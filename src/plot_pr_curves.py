
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def plot_pr_curve(
    features_df,
    saintq_df_raw,
    saintexpress_df_raw,
    go_cc_df,
    gt_source='bg_large'
):


    # Function to add missing preys for all baits in a dataset
    def add_missing_preys(df, features_df, prey_col):
        updated_df = pd.DataFrame()
        for bait in features_df['Bait'].unique():
            unique_preys = set(features_df[features_df['Bait'] == bait]['Prey'])
            existing_preys = set(df[df['Bait'] == bait][prey_col])
            missing_preys = unique_preys - existing_preys
            if missing_preys:
                print(f"Adding {len(missing_preys)} missing preys for bait {bait}.")
                missing_df = pd.DataFrame({
                    'Bait': [bait] * len(missing_preys),
                    prey_col: list(missing_preys),
                    'BFDR': [1.0] * len(missing_preys),
                    'AvgP': [0] * len(missing_preys)
                })
                updated_df = pd.concat([updated_df, missing_df], ignore_index=True)
            updated_df = pd.concat([updated_df, df[df['Bait'] == bait]], ignore_index=True)
        return updated_df

    saintq_df_raw_complete = add_missing_preys(saintq_df_raw, features_df, 'PreyGene')
    saintexpress_df_raw_complete = add_missing_preys(saintexpress_df_raw, features_df, 'PreyGene')


    # Normalize composite score globally
    features_df['composite_score_norm'] = MinMaxScaler().fit_transform(
        features_df[['composite_score']]
    ).flatten()

    gt_source = 'bg_large'
    # Load GO:CC ground truth interactions
    # go_cc_df = pd.read_csv('go_cc_interactions_large_benchmarking_bioid_dia.csv') # GOCC ground truth
    go_cc_df = pd.read_csv('biogrid_interactions_large_Benchmarking_BioID_DIA.csv') #BioGRID ground truth
    go_cc_dict = {bait: set(go_cc_df[go_cc_df['Bait'] == bait]['Prey']) for bait in features_df['Bait'].unique()}


    # New version based on selecting top-N preys
    def calculate_precision_recall_f1_topn(df, score_column, true_label_column, step_size=10):
        df_sorted = df.sort_values(score_column, ascending=False).reset_index(drop=True)
        precision, recall, f1_scores, threshold_dots = [], [], [], {}
        
        n_total_positives = df_sorted[true_label_column].sum()
        if n_total_positives == 0:
            print("Warning: no positive labels found!")

        topn_list = np.arange(step_size, len(df_sorted) + step_size, step_size)
        for topn in topn_list:
            selected = df_sorted.iloc[:topn]
            tp = selected[true_label_column].sum()
            fp = topn - tp
            fn = n_total_positives - tp

            precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_value = (2 * precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0

            precision.append(precision_value)
            recall.append(recall_value)
            f1_scores.append(f1_value)

            if topn in [100, 300]:  # Example: highlight top 100, 300, 500
                threshold_dots[topn] = (recall_value, precision_value)

        return np.array(recall), np.array(precision), f1_scores, threshold_dots

    # Function to calculate bootstrapped AUC
    def bootstrap_auc(recall, precision, n_bootstrap=100):
        bootstrapped_aucs = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(recall), len(recall))
            recall_resampled = recall[indices]
            precision_resampled = precision[indices]
            sorted_indices = np.argsort(recall_resampled)
            auc_value = auc(recall_resampled[sorted_indices], precision_resampled[sorted_indices])
            bootstrapped_aucs.append(auc_value)
        return np.mean(bootstrapped_aucs), np.std(bootstrapped_aucs)

    # Combine datasets across all baits for each method
    combined_pu_df, combined_comp_df = pd.DataFrame(), pd.DataFrame()

    combined_saintq_df, combined_saintexpress_df = pd.DataFrame(), pd.DataFrame()

    for bait in features_df['Bait'].unique():
        print(bait)
        bait_df = features_df[features_df['Bait'] == bait].copy()
        ms_preys = set(bait_df['Prey'])
        gold = go_cc_dict[bait] & ms_preys
        bait_df['true_label'] = bait_df['Prey'].apply(lambda x: 1 if x in gold else 0)
        combined_pu_df = pd.concat([combined_pu_df, bait_df], ignore_index=True)
        combined_comp_df = pd.concat([combined_comp_df, bait_df], ignore_index=True)

        saintq_df = saintq_df_raw_complete[saintq_df_raw_complete['Bait'] == bait].copy()
        ms_preys_q = set(saintq_df['PreyGene'])
        gold_q = go_cc_dict[bait] & ms_preys_q
        saintq_df['true_label'] = saintq_df['PreyGene'].apply(lambda x: 1 if x in gold_q else 0)
        combined_saintq_df = pd.concat([combined_saintq_df, saintq_df], ignore_index=True)

        saintexpress_df = saintexpress_df_raw_complete[saintexpress_df_raw_complete['Bait'] == bait].copy()
        ms_preys_ex = set(saintexpress_df['PreyGene'])
        gold_ex = go_cc_dict[bait] & ms_preys_ex
        saintexpress_df['true_label'] = saintexpress_df['PreyGene'].apply(lambda x: 1 if x in gold_ex else 0)
        combined_saintexpress_df = pd.concat([combined_saintexpress_df, saintexpress_df], ignore_index=True)


    recall_pu, precision_pu, f1_pu, dots_pu = calculate_precision_recall_f1_topn(combined_pu_df, 'predicted_probability', 'true_label')
    recall_comp, precision_comp, f1_comp, dots_comp = calculate_precision_recall_f1_topn(combined_comp_df, 'composite_score_norm', 'true_label')
    recall_q, precision_q, f1_q, dots_q = calculate_precision_recall_f1_topn(combined_saintq_df, 'AvgP', 'true_label')
    recall_ex, precision_ex, f1_ex, dots_ex = calculate_precision_recall_f1_topn(combined_saintexpress_df, 'AvgP', 'true_label')

    mean_f1_pu, mean_f1_comp, mean_f1_q, mean_f1_ex = np.mean(f1_pu), np.mean(f1_comp), np.mean(f1_q), np.mean(f1_ex)



    mean_auc_pu, std_auc_pu = bootstrap_auc(recall_pu, precision_pu)
    mean_auc_comp, std_auc_comp = bootstrap_auc(recall_comp, precision_comp)
    mean_auc_q, std_auc_q = bootstrap_auc(recall_q, precision_q)
    mean_auc_ex, std_auc_ex = bootstrap_auc(recall_ex, precision_ex)


    plt.figure(figsize=(8, 8))
    plt.plot(recall_pu, precision_pu, label=f'PU Learning\nAUC: {mean_auc_pu:.4f} ± {std_auc_pu:.4f}\nF1: {mean_f1_pu:.4f}', marker='o', markersize=2)
    plt.plot(recall_q, precision_q, label=f'SAINTq\nAUC: {mean_auc_q:.4f} ± {std_auc_q:.4f}\nF1: {mean_f1_q:.4f}', marker='o', markersize=2)
    plt.plot(recall_ex, precision_ex, label=f'SAINTexpress\nAUC: {mean_auc_ex:.4f} ± {std_auc_ex:.4f}\nF1: {mean_f1_ex:.4f}', marker='o', markersize=2)
    plt.plot(recall_comp, precision_comp, label=f'Composite\nAUC: {mean_auc_comp:.4f} ± {std_auc_comp:.4f}\nF1: {mean_f1_comp:.4f}', marker='o', markersize=2)


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision vs. Recall (All Baits Combined)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'results/Benchmarking BioID DIA/overall_precision_vs_recall_combined_{gt_source}.pdf', dpi=300)
    plt.close()

    # Compute AUC at top 100 and 500
    def auc_at_topn(df, score_col, true_col, topn):
        df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        df_top = df_sorted.iloc[:topn]
        recall, precision, _, _ = calculate_precision_recall_f1_topn(df_top, score_col, true_col, step_size=10)
        return auc(recall, precision)

    individual_features = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff', 
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio', 
        'zero_or_neg_fc',
    ]

    feature_pr_curves = []
    feature_auc_data = []

    for feat in individual_features:
        print(f"Processing feature: {feat}")
        # Drop NA to avoid issues
        feat_df = combined_pu_df.dropna(subset=[feat, 'true_label'])

        # Calculate PR for this feature
        recall_feat, precision_feat, f1_feat, dots_feat = calculate_precision_recall_f1_topn(
            feat_df, feat, 'true_label'
        )
        mean_auc_feat, std_auc_feat = bootstrap_auc(recall_feat, precision_feat)
        mean_f1_feat = np.mean(f1_feat)

        # Store for PR plot
        feature_pr_curves.append((recall_feat, precision_feat, feat, mean_auc_feat, std_auc_feat, mean_f1_feat))

        # AUC@Top300
        auc_feat_top300 = auc_at_topn(feat_df, feat, 'true_label', 300)

        # Store AUC values
        feature_auc_data.append({
            'Method': feat,
            'Total AUC': mean_auc_feat,
            'AUC@Top300': auc_feat_top300
        })

    # ---- Plot PR curves (including features) ---- #

    plt.figure(figsize=(10, 10))
    plt.plot(recall_pu, precision_pu, label=f'PU Learning\nAUC: {mean_auc_pu:.4f} ± {std_auc_pu:.4f}\nF1: {mean_f1_pu:.4f}', marker='o', markersize=2)
    plt.plot(recall_q, precision_q, label=f'SAINTq\nAUC: {mean_auc_q:.4f} ± {std_auc_q:.4f}\nF1: {mean_f1_q:.4f}', marker='o', markersize=2)
    plt.plot(recall_ex, precision_ex, label=f'SAINTexpress\nAUC: {mean_auc_ex:.4f} ± {std_auc_ex:.4f}\nF1: {mean_f1_ex:.4f}', marker='o', markersize=2)
    plt.plot(recall_comp, precision_comp, label=f'Composite\nAUC: {mean_auc_comp:.4f} ± {std_auc_comp:.4f}\nF1: {mean_f1_comp:.4f}', marker='o', markersize=2)

    # Plot individual features
    for r, p, label, auc_val, std_val, f1_val in feature_pr_curves:
        plt.plot(r, p, label=f'{label}\nAUC: {auc_val:.4f} ± {std_val:.4f}\nF1: {f1_val:.4f}', linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve Including Individual Features')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/overall_precision_vs_recall_combined_{gt_source}_with_features.pdf', dpi=300)
    plt.close()




    auc_data = {
        'Method': ['PU Learning', 'SAINTq', 'SAINTexpress', 'Composite score'],
        'Total AUC': [mean_auc_pu, mean_auc_q, mean_auc_ex, mean_auc_comp],
        'AUC@Top300': [
            auc_at_topn(combined_pu_df, 'predicted_probability', 'true_label', 300),
            auc_at_topn(combined_saintq_df, 'AvgP', 'true_label', 300),
            auc_at_topn(combined_saintexpress_df, 'AvgP', 'true_label', 300),
            auc_at_topn(combined_comp_df, 'composite_score_norm', 'true_label', 300),
        ]
    }

    auc_df = pd.DataFrame(auc_data)

    # Melt for seaborn plotting
    auc_df_melted = auc_df.melt(id_vars='Method', var_name='AUC Type', value_name='AUC')

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=auc_df_melted, x='AUC Type', y='AUC', hue='Method', palette='muted')
    # plt.title('AUC Comparison: Total, Top 100, Top 500')

    # Axis labels
    plt.ylabel('AUC Value', fontsize=26)
    plt.xlabel('', fontsize=26) 

    # Tick labels
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # Legend
    # plt.legend(title='Method', ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=18)
    plt.legend().remove()
    # plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/auc_barplot_combined_{gt_source}.pdf', dpi=300)
    plt.close()

    # ---- Add to AUC bar plot ---- #
    feature_auc_df = pd.DataFrame(feature_auc_data)
    auc_df_extended = pd.concat([auc_df, feature_auc_df], ignore_index=True)

    # Melt for seaborn
    auc_df_melted = auc_df_extended.melt(id_vars='Method', var_name='AUC Type', value_name='AUC')

    # Bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=auc_df_melted, x='AUC Type', y='AUC', hue='Method', palette='tab20')
    plt.ylabel('AUC Value', fontsize=26)
    plt.xlabel('', fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(title='Method / Feature', ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/auc_barplot_combined_{gt_source}_with_features.pdf', dpi=300)
    plt.close()

    # Define FDR bins
    fdr_bins = [0, 0.01, 0.05]
    fdr_labels = ['0-0.01', '0.01-0.05']

    # Function to bin FDR and calculate precision
    def compute_precision_by_fdr_bin(df, score_col, label_col, method_name):
        df = df.copy()
        df['FDR_bin'] = pd.cut(df[score_col], bins=fdr_bins, labels=fdr_labels, include_lowest=True)
        grouped = df.groupby('FDR_bin')[label_col].agg(['sum', 'count']).reset_index()
        grouped['Precision'] = grouped['sum'] / grouped['count']
        grouped['Method'] = method_name
        return grouped[['FDR_bin', 'Precision', 'Method']]

    # Apply to each method
    df_pu = compute_precision_by_fdr_bin(combined_pu_df, 'FDR', 'true_label', 'PU Learning')
    df_q = compute_precision_by_fdr_bin(combined_saintq_df, 'BFDR', 'true_label', 'SAINTq')
    df_ex = compute_precision_by_fdr_bin(combined_saintexpress_df, 'BFDR', 'true_label', 'SAINTexpress')
    # df_comp = compute_precision_by_fdr_bin(combined_comp_df, 'FDR', 'true_label', 'Composite')

    # Combine all results
    precision_bins_df = pd.concat([df_pu, df_q, df_ex], ignore_index=True)

    # Plot
    plt.figure(figsize=(7, 5))
    sns.barplot(data=precision_bins_df, x='FDR_bin', y='Precision', hue='Method', palette='muted', width=0.6)
    # plt.title('Precision by FDR Bin for Each Method')
    plt.ylabel('Precision', fontsize=20)
    plt.xlabel('FDR Bin', fontsize=20)
    # plt.ylim(0, 1.05)
    # plt.legend(title='Method')
    plt.legend().remove()
    # Tick labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/precision_by_fdr_bin_{gt_source}.pdf', dpi=300)
    plt.close()




    # Function to bin FDR and count true positives
    def compute_true_positives_by_fdr_bin(df, score_col, label_col, method_name):
        df = df.copy()
        df['FDR_bin'] = pd.cut(df[score_col], bins=fdr_bins, labels=fdr_labels, include_lowest=True)
        grouped = df.groupby('FDR_bin')[label_col].sum().reset_index()
        grouped['Method'] = method_name
        grouped.rename(columns={label_col: 'True Positives'}, inplace=True)
        return grouped

    # Apply to each method
    df_pu_tp = compute_true_positives_by_fdr_bin(combined_pu_df, 'FDR', 'true_label', 'PU Learning')
    df_q_tp = compute_true_positives_by_fdr_bin(combined_saintq_df, 'BFDR', 'true_label', 'SAINTq')
    df_ex_tp = compute_true_positives_by_fdr_bin(combined_saintexpress_df, 'BFDR', 'true_label', 'SAINTexpress')
    # df_comp_tp = compute_true_positives_by_fdr_bin(combined_comp_df, 'FDR', 'true_label', 'Composite')

    # Combine all results
    tp_bins_df = pd.concat([df_pu_tp, df_q_tp, df_ex_tp], ignore_index=True)

    # Plot
    plt.figure(figsize=(7, 5))
    sns.barplot(data=tp_bins_df, x='FDR_bin', y='True Positives', hue='Method', palette='muted', width=0.6)
    # plt.title('Number of True Positives by FDR Bin for Each Method')
    plt.ylabel('Number of True Positives', fontsize=20)
    plt.xlabel('FDR Bin', fontsize=20)
    # Tick labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.legend(title='Method')
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/true_positives_by_fdr_bin_{gt_source}.pdf', dpi=300)
    plt.close()



    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot Precision
    sns.barplot(data=precision_bins_df, x='FDR_bin', y='Precision', hue='Method', ax=axes[0], palette='muted')
    axes[0].set_title('Precision by FDR Bin', fontsize=22)
    axes[0].set_ylabel('Precision', fontsize=20)
    axes[0].set_xlabel('', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].legend(title='Method', loc=(1.05, 0.95), fontsize=18, title_fontsize=18)

    # Plot True Positives
    sns.barplot(data=tp_bins_df, x='FDR_bin', y='True Positives', hue='Method', ax=axes[1], palette='muted')
    axes[1].set_title('Number of True Positives by FDR Bin', fontsize=22)
    axes[1].set_ylabel('True Positives', fontsize=20)
    axes[1].set_xlabel('FDR Bin', fontsize=18)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].legend_.remove()  # Remove duplicate legend

    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/precision_and_tp_by_fdr_bin_{gt_source}.pdf', dpi=300)
    plt.close()


    # Plot F1 score distributions
    plt.figure(figsize=(8, 6))

    f1_data = [f1_pu, f1_q, f1_ex, f1_comp]
    method_names = ['PU Learning', 'SAINTq', 'SAINTexpress', 'Composite']

    plt.boxplot(f1_data, labels=method_names, patch_artist=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))

    plt.ylabel('F1 Score')
    plt.title('Distribution of F1 Scores Across Thresholds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/f1_score_distribution_boxplot_{gt_source}.pdf', dpi=300)
    plt.close()


    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Score Distributions by True/False Label (All Baits Combined)', fontsize=14)

    # PU Learning
    axes[0, 0].hist(
        [combined_pu_df[combined_pu_df['true_label'] == 1]['predicted_probability'],
        combined_pu_df[combined_pu_df['true_label'] == 0]['predicted_probability']],
        bins=50, stacked=True, density=True, label=['True', 'False'], color=['navy', 'skyblue']
    )
    axes[0, 0].set_title('PU Learning')
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # SAINTq
    axes[0, 1].hist(
        [combined_saintq_df[combined_saintq_df['true_label'] == 1]['AvgP'],
        combined_saintq_df[combined_saintq_df['true_label'] == 0]['AvgP']],
        bins=50, stacked=True, density=True, label=['True', 'False'], color=['darkgreen', 'lightgreen']
    )
    axes[0, 1].set_title('SAINTq')
    axes[0, 1].set_xlabel('AvgP')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()

    # SAINTexpress
    axes[1, 0].hist(
        [combined_saintexpress_df[combined_saintexpress_df['true_label'] == 1]['AvgP'],
        combined_saintexpress_df[combined_saintexpress_df['true_label'] == 0]['AvgP']],
        bins=50, stacked=True, density=True, label=['True', 'False'], color=['darkred', 'lightcoral']
    )
    axes[1, 0].set_title('SAINTexpress')
    axes[1, 0].set_xlabel('AvgP')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()

    # Composite
    axes[1, 1].hist(
        [combined_comp_df[combined_comp_df['true_label'] == 1]['composite_score_norm'],
        combined_comp_df[combined_comp_df['true_label'] == 0]['composite_score_norm']],
        bins=50, stacked=True, density=True, label=['True', 'False'], color=['goldenrod', 'khaki']
    )
    axes[1, 1].set_title('Composite Score')
    axes[1, 1].set_xlabel('Normalized Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'results/Benchmarking BioID DIA/score_distributions_hist_all_baits_combined_{gt_source}.pdf', dpi=300)
    plt.close()




    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Score Distributions by True/False Label (All Baits Combined)', fontsize=14)

    # PU Learning
    sns.kdeplot(
        data=combined_pu_df, x='predicted_probability', hue='true_label',
        fill=True, common_norm=False, palette={1: 'navy', 0: 'skyblue'}, ax=axes[0, 0]
    )
    axes[0, 0].set_title('PU Learning')
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend(title='Label', labels=['False', 'True'])

    # SAINTq
    sns.kdeplot(
        data=combined_saintq_df, x='AvgP', hue='true_label',
        fill=True, common_norm=False, palette={1: 'darkgreen', 0: 'lightgreen'}, ax=axes[0, 1]
    )
    axes[0, 1].set_title('SAINTq')
    axes[0, 1].set_xlabel('AvgP')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend(title='Label', labels=['False', 'True'])

    # SAINTexpress
    sns.kdeplot(
        data=combined_saintexpress_df, x='AvgP', hue='true_label',
        fill=True, common_norm=False, palette={1: 'darkred', 0: 'lightcoral'}, ax=axes[1, 0]
    )
    axes[1, 0].set_title('SAINTexpress')
    axes[1, 0].set_xlabel('AvgP')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend(title='Label', labels=['False', 'True'])

    # Composite Score
    sns.kdeplot(
        data=combined_comp_df, x='composite_score_norm', hue='true_label',
        fill=True, common_norm=False, palette={1: 'goldenrod', 0: 'khaki'}, ax=axes[1, 1]
    )
    axes[1, 1].set_title('Composite Score')
    axes[1, 1].set_xlabel('Normalized Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend(title='Label', labels=['False', 'True'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'results/Benchmarking BioID DIA/score_distributions_density_all_baits_combined_{gt_source}.pdf', dpi=300)
    plt.close()


    import pandas as pd
    import seaborn as sns

    # Prepare data
    pu_df = combined_pu_df.copy()
    pu_df['Method'] = 'PU Learning'
    pu_df['Score'] = pu_df['predicted_probability']

    saintq_df = combined_saintq_df.copy()
    saintq_df['Method'] = 'SAINTq'
    saintq_df['Score'] = saintq_df['AvgP']

    saintex_df = combined_saintexpress_df.copy()
    saintex_df['Method'] = 'SAINTexpress'
    saintex_df['Score'] = saintex_df['AvgP']

    comp_df = combined_comp_df.copy()
    comp_df['Method'] = 'Composite score'
    comp_df['Score'] = comp_df['composite_score_norm']

    # Combine all into one DataFrame
    plot_df = pd.concat([pu_df, saintq_df, saintex_df, comp_df], ignore_index=True)

    #Define palette: 0 = False (red), 1 = True (blue)
    hue_order = [0, 1]
    palette = {
        0: 'lightcoral',     # False → red
        1: 'dodgerblue'      # True → blue
    }

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=plot_df,
        x='Method',
        y='Score',
        hue='true_label',
        palette=palette,
        hue_order=hue_order,
        fliersize=2  # Smaller outlier marker
    )

    plt.title('Score Distributions by Method and Label (Boxplot)')
    plt.ylabel('Score / Confidence')
    plt.xlabel('')

    # Place legend outside plot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, ['False', 'True'],
        title='True Label',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/boxplot_score_distributions_{gt_source}.pdf', dpi=300, bbox_inches='tight')
    plt.close()




    # Prepare data
    pu_df = combined_pu_df.copy()
    pu_df['Method'] = 'PU Learning'
    pu_df['Score'] = pu_df['predicted_probability']

    saintq_df = combined_saintq_df.copy()
    saintq_df['Method'] = 'SAINTq'
    saintq_df['Score'] = saintq_df['AvgP']

    saintex_df = combined_saintexpress_df.copy()
    saintex_df['Method'] = 'SAINTexpress'
    saintex_df['Score'] = saintex_df['AvgP']

    comp_df = combined_comp_df.copy()
    comp_df['Method'] = 'Composite score'
    comp_df['Score'] = comp_df['composite_score_norm']

    # Combine all into one DataFrame
    plot_df = pd.concat([pu_df, saintq_df, saintex_df, comp_df], ignore_index=True)

    # Define palette
    hue_order = [0, 1]
    palette = {
        0: 'lightcoral',
        1: 'dodgerblue'
    }

    # Start plot
    plt.figure(figsize=(10, 6))

    # Violin plot
    sns.violinplot(
        data=plot_df,
        x='Method',
        y='Score',
        hue='true_label',
        palette=palette,
        hue_order=hue_order,
        cut=0,
        scale='width',
        inner='quartile',
        dodge=True
    )

    # # Overlay dots
    # sns.stripplot(
    #     data=plot_df,
    #     x='Method',
    #     y='Score',
    #     hue='true_label',
    #     palette=palette,
    #     hue_order=hue_order,
    #     dodge=True,
    #     jitter=0.2,
    #     alpha=0.3,
    #     marker='o',
    #     linewidth=0.5,
    #     edgecolor='gray'
    # )

    # Fix duplicated legend
    handles, labels = plt.gca().get_legend_handles_labels()
    n = len(hue_order)
    plt.legend(
        handles[:n],
        ['False', 'True'],
        title='True Label',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.title('Score Distributions by Method and Label (Violin + Dots)')
    plt.ylabel('Score / Confidence')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/violinplot_with_dots_score_distributions_{gt_source}.pdf', dpi=300, bbox_inches='tight')
    plt.close()



    # True Positives vs Score (All Baits Combined)
    tp_results_score = []
    tp_results_fdr = []

    # PU Learning
    features_df = features_df.copy()
    features_df['true_label'] = features_df.apply(lambda row: 1 if row['Prey'] in go_cc_dict.get(row['Bait'], set()) else 0, axis=1)

    # PU: probability vs TP
    sorted_probs = features_df.sort_values("predicted_probability", ascending=False)
    cumulative_tp = sorted_probs['true_label'].cumsum()
    tp_results_score.append(('PU Learning', sorted_probs['predicted_probability'].values, cumulative_tp.values))

    # PU: FDR vs TP
    sorted_fdr = features_df.sort_values("FDR")
    cumulative_tp_fdr = sorted_fdr['true_label'].cumsum()
    tp_results_fdr.append(('PU Learning', sorted_fdr['FDR'].values, cumulative_tp_fdr.values))

    # Composite Score (normalized)
    composite_scores = features_df['composite_score']
    scaled_scores = (composite_scores - composite_scores.min()) / (composite_scores.max() - composite_scores.min())
    features_df['composite_score_scaled'] = scaled_scores
    sorted_composite = features_df.sort_values("composite_score_scaled", ascending=False)
    cumulative_tp_comp = sorted_composite['true_label'].cumsum()
    tp_results_score.append(('Composite Score', sorted_composite['composite_score_scaled'].values, cumulative_tp_comp.values))

    # SAINTq (use combined dataframe)
    combined_saintq_df = combined_saintq_df.copy()
    combined_saintq_df['true_label'] = combined_saintq_df.apply(lambda row: 1 if row['PreyGene'] in go_cc_dict.get(row['Bait'], set()) else 0, axis=1)
    sorted_probs = combined_saintq_df.sort_values("AvgP", ascending=False)
    cumulative_tp = sorted_probs['true_label'].cumsum()
    tp_results_score.append(('SAINTq', sorted_probs['AvgP'].values, cumulative_tp.values))

    sorted_fdr = combined_saintq_df.sort_values("BFDR")
    cumulative_tp_fdr = sorted_fdr['true_label'].cumsum()
    tp_results_fdr.append(('SAINTq', sorted_fdr['BFDR'].values, cumulative_tp_fdr.values))

    # SAINTexpress (use combined dataframe)
    combined_saintexpress_df = combined_saintexpress_df.copy()
    combined_saintexpress_df['true_label'] = combined_saintexpress_df.apply(lambda row: 1 if row['PreyGene'] in go_cc_dict.get(row['Bait'], set()) else 0, axis=1)
    sorted_probs = combined_saintexpress_df.sort_values("AvgP", ascending=False)
    cumulative_tp = sorted_probs['true_label'].cumsum()
    tp_results_score.append(('SAINTexpress', sorted_probs['AvgP'].values, cumulative_tp.values))

    sorted_fdr = combined_saintexpress_df.sort_values("BFDR")
    cumulative_tp_fdr = sorted_fdr['true_label'].cumsum()
    tp_results_fdr.append(('SAINTexpress', sorted_fdr['BFDR'].values, cumulative_tp_fdr.values))

    # Plot TP vs Score (Probability or Composite)
    plt.figure(figsize=(10, 6))
    colors = {
        'PU Learning': '#1f77b4',
        'SAINTq': '#ff7f0e',
        'SAINTexpress': '#2ca02c',
        'Composite Score': 'red'
    }
    for method, scores, tps in tp_results_score:
        plt.plot(scores, tps, label=method, color=colors.get(method, 'gray'))
    plt.xlabel('Score (Probability / AvgP)')
    plt.ylabel('Cumulative True Positives')
    plt.title('True Positives vs Score (All Baits Combined)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/true_positives_vs_score_all_baits_{gt_source}.pdf', dpi=300)
    plt.close()

    # Plot TP vs FDR/BFDR
    plt.figure(figsize=(10, 6))
    for method, fdrs, tps in tp_results_fdr:
        plt.plot(fdrs, tps, label=method, color=colors.get(method, 'gray'))
    plt.xlabel('FDR / BFDR')
    plt.ylabel('Cumulative True Positives')
    plt.title('True Positives vs FDR (All Baits Combined)')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/true_positives_vs_fdr_all_baits_{gt_source}.pdf', dpi=300)
    plt.close()




















