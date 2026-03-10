def plot_pr_curve(
    features_df,
    saintq_df_raw,
    saintexpress_df_raw,
    limma_df_raw,              # limma table
    go_cc_df,
    gt_source='bg_large',
    outdir='results'
):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import auc
    from sklearn.preprocessing import MinMaxScaler

    os.makedirs(outdir, exist_ok=True)

    # Function to add missing preys for all baits in a dataset
    def add_missing_preys(df, features_df, prey_col, fill_cols=None):
        updated_df = pd.DataFrame()
        fill_cols = fill_cols or {}
        for bait in features_df['Bait'].unique():
            unique_preys = set(features_df[features_df['Bait'] == bait]['Prey'])
            existing_preys = set(df[df['Bait'] == bait][prey_col])
            missing_preys = unique_preys - existing_preys
            if missing_preys:
                base = {'Bait': [bait] * len(missing_preys), prey_col: list(missing_preys)}
                base.update({k: [v] * len(missing_preys) for (k, v) in fill_cols.items()})
                missing_df = pd.DataFrame(base)
                updated_df = pd.concat([updated_df, missing_df], ignore_index=True)
            updated_df = pd.concat([updated_df, df[df['Bait'] == bait]], ignore_index=True)
        return updated_df

    # SAINT tables (fill BFDR=1, AvgP=0 for missing)
    saintq_df_raw_complete = add_missing_preys(
        saintq_df_raw, features_df, 'PreyGene',
        fill_cols={'BFDR': 1.0, 'AvgP': 0.0}
    )
    saintexpress_df_raw_complete = add_missing_preys(
        saintexpress_df_raw, features_df, 'PreyGene',
        fill_cols={'BFDR': 1.0, 'AvgP': 0.0}
    )

    # limma table (fill adj.P.Val=1.0 for missing) → score = 1 - adj.P.Val
    limma_df_raw_complete = add_missing_preys(
        limma_df_raw.rename(columns={'Protein': 'Prey'}),  # if needed
        features_df, 'Prey',
        fill_cols={'adj.P.Val': 1.0}
    )
    limma_df_raw_complete['adj.P.Val'] = limma_df_raw_complete['adj.P.Val'].astype(float).clip(0.0, 1.0)
    limma_df_raw_complete['limma_score'] = 1.0 - limma_df_raw_complete['adj.P.Val']

    # Normalize heuristic score globally
    features_df = features_df.copy()
    features_df['heuristic_score_norm'] = MinMaxScaler().fit_transform(
        features_df[['heuristic_score']]
    ).flatten()

    # Ground truth dict: use provided go_cc_df + gt_source
    go_cc_dict = {
        bait: set(go_cc_df[go_cc_df['Bait'] == bait]['Prey'])
        for bait in features_df['Bait'].unique()
    }

    # PR helpers
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
            if topn in [100, 300]:
                threshold_dots[topn] = (recall_value, precision_value)
        return np.array(recall), np.array(precision), f1_scores, threshold_dots

    def bootstrap_auc(recall, precision, n_bootstrap=100):
        if len(recall) == 0:
            return np.nan, np.nan
        boot = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(recall), len(recall))
            r = recall[idx]; p = precision[idx]
            order = np.argsort(r)
            boot.append(auc(r[order], p[order]))
        return float(np.mean(boot)), float(np.std(boot))

    # Combine datasets across all baits for each method
    combined_pu_df, combined_heuristic_df = pd.DataFrame(), pd.DataFrame()
    combined_saintq_df, combined_saintexpress_df, combined_limma_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for bait in features_df['Bait'].unique():
        bait_df = features_df[features_df['Bait'] == bait].copy()
        ms_preys = set(bait_df['Prey'])
        gold = go_cc_dict[bait] & ms_preys
        bait_df['true_label'] = bait_df['Prey'].apply(lambda x: 1 if x in gold else 0)
        combined_pu_df = pd.concat([combined_pu_df, bait_df], ignore_index=True)
        combined_heuristic_df = pd.concat([combined_heuristic_df, bait_df], ignore_index=True)

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

        limma_df = limma_df_raw_complete[limma_df_raw_complete['Bait'] == bait].copy()
        ms_preys_l = set(limma_df['Prey'])
        gold_l = go_cc_dict[bait] & ms_preys_l
        limma_df['true_label'] = limma_df['Prey'].apply(lambda x: 1 if x in gold_l else 0)
        combined_limma_df = pd.concat([combined_limma_df, limma_df], ignore_index=True)

    # Compute PR for all methods
    recall_pu,    precision_pu,    f1_pu,    _ = calculate_precision_recall_f1_topn(combined_pu_df, 'predicted_probability', 'true_label')
    recall_heuristic,  precision_heuristic,  f1_heuristic,  _ = calculate_precision_recall_f1_topn(combined_heuristic_df, 'heuristic_score_norm', 'true_label')
    recall_q,     precision_q,     f1_q,     _ = calculate_precision_recall_f1_topn(combined_saintq_df, 'AvgP', 'true_label')
    recall_ex,    precision_ex,    f1_ex,    _ = calculate_precision_recall_f1_topn(combined_saintexpress_df, 'AvgP', 'true_label')
    recall_limma, precision_limma, f1_limma, _ = calculate_precision_recall_f1_topn(combined_limma_df, 'limma_score', 'true_label')

    mean_f1_pu, mean_f1_heuristic = np.mean(f1_pu), np.mean(f1_heuristic)
    mean_f1_q,  mean_f1_ex  = np.mean(f1_q),  np.mean(f1_ex)
    mean_f1_limma = np.mean(f1_limma)

    mean_auc_pu,    std_auc_pu    = bootstrap_auc(recall_pu,    precision_pu)
    mean_auc_heuristic,  std_auc_heuristic  = bootstrap_auc(recall_heuristic,  precision_heuristic)
    mean_auc_q,     std_auc_q     = bootstrap_auc(recall_q,     precision_q)
    mean_auc_ex,    std_auc_ex    = bootstrap_auc(recall_ex,    precision_ex)
    mean_auc_limma, std_auc_limma = bootstrap_auc(recall_limma, precision_limma)

    # Helper for AUC@TopN
    def auc_at_topn(df, score_col, true_col, topn):
        df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        df_top = df_sorted.iloc[:topn]
        r, p, _, _ = calculate_precision_recall_f1_topn(df_top, score_col, true_col, step_size=10)
        return auc(r, p)

    # ---- Main PR plot (methods + heuristic) ----
    plt.figure(figsize=(8, 8))
    plt.plot(recall_pu,    precision_pu,    label=f'whisper\nAUC: {mean_auc_pu:.4f} ± {std_auc_pu:.4f}\nF1: {mean_f1_pu:.4f}', marker='o', markersize=2)
    plt.plot(recall_q,     precision_q,     label=f'SAINTq\nAUC: {mean_auc_q:.4f} ± {std_auc_q:.4f}\nF1: {mean_f1_q:.4f}', marker='o', markersize=2)
    plt.plot(recall_ex,    precision_ex,    label=f'SAINTexpress\nAUC: {mean_auc_ex:.4f} ± {std_auc_ex:.4f}\nF1: {mean_f1_ex:.4f}', marker='o', markersize=2)
    plt.plot(recall_limma, precision_limma, label=f'limma\nAUC: {mean_auc_limma:.4f} ± {std_auc_limma:.4f}\nF1: {mean_f1_limma:.4f}', marker='o', markersize=2)
    plt.plot(recall_heuristic,  precision_heuristic,  label=f'heuristic\nAUC: {mean_auc_heuristic:.4f} ± {std_auc_heuristic:.4f}\nF1: {mean_f1_heuristic:.4f}', marker='o', markersize=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision vs. Recall (All Baits Combined)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'overall_precision_vs_recall_combined_{gt_source}.pdf'), dpi=300)
    plt.close()

    # Individual feature baselines
    individual_features = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff',
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
        'zero_or_neg_fc',
    ]
    feature_pr_curves, feature_auc_data = [], []
    for feat in individual_features:
        feat_df = combined_pu_df.dropna(subset=[feat, 'true_label'])
        r_f, p_f, f1_f, _ = calculate_precision_recall_f1_topn(feat_df, feat, 'true_label')
        auc_f_mean, auc_f_std = bootstrap_auc(r_f, p_f)
        f1_f_mean = np.mean(f1_f)
        feature_pr_curves.append((r_f, p_f, feat, auc_f_mean, auc_f_std, f1_f_mean))
        auc_feat_top300 = auc_at_topn(feat_df, feat, 'true_label', 300)
        feature_auc_data.append({'Method': feat, 'Total AUC': auc_f_mean, 'AUC@Top300': auc_feat_top300})

    # ---- PR plot including features ----
    plt.figure(figsize=(10, 10))
    plt.plot(recall_pu,    precision_pu,    label=f'whisper\nAUC: {mean_auc_pu:.4f} ± {std_auc_pu:.4f}\nF1: {mean_f1_pu:.4f}', marker='o', markersize=2)
    plt.plot(recall_q,     precision_q,     label=f'SAINTq\nAUC: {mean_auc_q:.4f} ± {std_auc_q:.4f}\nF1: {mean_f1_q:.4f}', marker='o', markersize=2)
    plt.plot(recall_ex,    precision_ex,    label=f'SAINTexpress\nAUC: {mean_auc_ex:.4f} ± {std_auc_ex:.4f}\nF1: {mean_f1_ex:.4f}', marker='o', markersize=2)
    plt.plot(recall_limma, precision_limma, label=f'limma\nAUC: {mean_auc_limma:.4f} ± {std_auc_limma:.4f}\nF1: {mean_f1_limma:.4f}', marker='o', markersize=2)
    plt.plot(recall_heuristic,  precision_heuristic,  label=f'heuristic\nAUC: {mean_auc_heuristic:.4f} ± {std_auc_heuristic:.4f}\nF1: {mean_f1_heuristic:.4f}', marker='o', markersize=2)

    for r, p, label, auc_val, std_val, f1_val in feature_pr_curves:
        plt.plot(r, p, label=f'{label}\nAUC: {auc_val:.4f} ± {std_val:.4f}\nF1: {f1_val:.4f}', linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve Including Individual Features')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'overall_precision_vs_recall_combined_{gt_source}_with_features.pdf'), dpi=300)
    plt.close()

    # ---- AUC barplots (incl. limma) ----
    auc_df = pd.DataFrame({
        'Method': ['whisper', 'SAINTq', 'SAINTexpress', 'limma', 'heuristic score'],
        'Total AUC': [mean_auc_pu, mean_auc_q, mean_auc_ex, mean_auc_limma, mean_auc_heuristic],
        'AUC@Top300': [
            auc_at_topn(combined_pu_df, 'predicted_probability', 'true_label', 300),
            auc_at_topn(combined_saintq_df, 'AvgP', 'true_label', 300),
            auc_at_topn(combined_saintexpress_df, 'AvgP', 'true_label', 300),
            auc_at_topn(combined_limma_df, 'limma_score', 'true_label', 300),
            auc_at_topn(combined_heuristic_df, 'heuristic_score_norm', 'true_label', 300),
        ]
    })
    auc_df_melted = auc_df.melt(id_vars='Method', var_name='AUC Type', value_name='AUC')

    plt.figure(figsize=(12, 8))
    sns.barplot(data=auc_df_melted, x='AUC Type', y='AUC', hue='Method', palette='muted')
    plt.ylabel('AUC Value', fontsize=26)
    plt.xlabel('', fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'auc_barplot_combined_{gt_source}.pdf'), dpi=300)
    plt.close()

    # ---- Extended barplot with features ----
    feature_auc_df = pd.DataFrame(feature_auc_data)
    auc_df_extended = pd.concat([auc_df, feature_auc_df], ignore_index=True)
    auc_df_melted_ext = auc_df_extended.melt(id_vars='Method', var_name='AUC Type', value_name='AUC')

    plt.figure(figsize=(14, 8))
    sns.barplot(data=auc_df_melted_ext, x='AUC Type', y='AUC', hue='Method', palette='tab20')
    plt.ylabel('AUC Value', fontsize=26)
    plt.xlabel('', fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(title='Method / Feature', ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.45), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'auc_barplot_combined_{gt_source}_with_features.pdf'), dpi=300)
    plt.close()