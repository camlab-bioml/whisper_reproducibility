import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def recovery_overlap(
    features_df,
    saintq_df_raw,
    saintexpress_df_raw,
    go_cc_df,
    gt_source='bg_large'
):



    # Ground truth: GO-based interactions
    # go_cc_df = pd.read_csv('go_cc_interactions_large_benchmarking_bioid_dia.csv')
    go_cc_df = pd.read_csv('biogrid_interactions_large_benchmarking_bioid_dia.csv')
    identified_preys = set(features_df['Prey'].unique())
    go_cc_df = go_cc_df[go_cc_df['Prey'].isin(identified_preys)]
    go_cc_dict = {bait: set(go_cc_df[go_cc_df['Bait'] == bait]['Prey']) for bait in features_df['Bait'].unique()}

    # Preprocess SAINT data to include missing preys
    def add_missing_preys(df, features_df, prey_col):
        updated_df = pd.DataFrame()
        for bait in features_df['Bait'].unique():
            unique_preys = set(features_df[features_df['Bait'] == bait]['Prey'])
            existing_preys = set(df[df['Bait'] == bait][prey_col])
            missing_preys = unique_preys - existing_preys
            if missing_preys:
                missing_df = pd.DataFrame({
                    'Bait': [bait] * len(missing_preys),
                    prey_col: list(missing_preys),
                    'BFDR': [1.0] * len(missing_preys),
                    'AvgP': [0] * len(missing_preys)
                })
                updated_df = pd.concat([updated_df, missing_df], ignore_index=True)
            updated_df = pd.concat([updated_df, df[df['Bait'] == bait]], ignore_index=True)
        return updated_df

    saintq_df = add_missing_preys(saintq_df_raw, features_df, 'PreyGene')
    saintexpress_df = add_missing_preys(saintexpress_df_raw, features_df, 'PreyGene')

    # Clean method-specific dataframes
    def prepare_method_df(df, bait_col, prey_col, score_col, method_name):
        df_clean = df[[bait_col, prey_col, score_col]].copy()
        df_clean.columns = ['Bait', 'Prey', 'Score']
        df_clean['Method'] = method_name
        return df_clean

    # Apply fixes for each method
    df_pu = features_df[['Bait', 'Prey', 'predicted_probability']].copy()
    df_pu.columns = ['Bait', 'Prey', 'Score']
    df_pu['Method'] = 'whisper'

    df_q = prepare_method_df(saintq_df, 'Bait', 'PreyGene', 'AvgP', 'SAINTq')
    df_ex = prepare_method_df(saintexpress_df, 'Bait', 'PreyGene', 'AvgP', 'SAINTexpress')

    # Normalize composite score
    from sklearn.preprocessing import MinMaxScaler
    # features_df['composite_score_norm'] = features_df.groupby('Bait')['composite_score'].transform(
    #     lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
    # )
    # features_df['composite_score_norm'] = features_df['composite_score']

    # Normalize composite score globally
    features_df['composite_score_norm'] = MinMaxScaler().fit_transform(
        features_df[['composite_score']]
    ).flatten()

    df_comp = features_df[['Bait', 'Prey', 'composite_score_norm']].copy()
    df_comp.columns = ['Bait', 'Prey', 'Score']
    df_comp['Method'] = 'Composite score'

    # Combine all methods
    df_all = pd.concat([df_pu, df_q, df_ex, df_comp], ignore_index=True)

    # Evaluate overlap across top N interactions
    step = 10
    max_n = 2000
    top_ks = np.arange(step, max_n + step, step)

    colors = {
        'whisper': '#1f77b4',
        'SAINTq': '#ff7f0e',
        'SAINTexpress': '#2ca02c',
        'Composite score': 'crimson'
    }

    plt.figure(figsize=(8, 6))

    for method in df_all['Method'].unique():
        df_method = df_all[df_all['Method'] == method].copy()
        df_method = df_method.sort_values('Score', ascending=False).reset_index(drop=True)

        overlap_rates = []
        for k in top_ks:
            top_k = df_method.iloc[:k]
            tp_count = 0
            for _, row in top_k.iterrows():
                bait, prey = row['Bait'], row['Prey']
                if prey in go_cc_dict.get(bait, set()):
                    tp_count += 1
            rate = (tp_count / k) * 100
            overlap_rates.append(rate)

        plt.plot(top_ks, overlap_rates, label=method, color=colors.get(method, 'gray'), marker='o', markersize=3)

    plt.xlabel('Number of Interactions (Top N)', fontsize=22)
    plt.ylabel('Database Overlap % (GO:CC)', fontsize=22)
    # plt.title('Database Overlap vs. Number of Interactions')
    # plt.grid(True, linestyle='--', alpha=0.5)

    # Tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/overlap_vs_interactions_{gt_source}.pdf', dpi=300)
    plt.close()

    # --- Add individual features ---
    individual_features = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff', 
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio', 
        'zero_or_neg_fc'
    ]

    linestyles = ['dashed'] * len(individual_features)
    feature_colors = {
        'log_fold_change': 'gray',
        'snr': 'black',
        'mean_diff': 'darkblue',
        'median_diff': 'darkgreen',
        'replicate_fold_change_sd': 'darkred',
        'bait_cv': 'darkorange',
        'bait_control_sd_ratio': 'darkviolet',
        'zero_or_neg_fc': 'brown'
    }

    plt.figure(figsize=(8, 6))

    # Plot main methods
    for method in df_all['Method'].unique():
        df_method = df_all[df_all['Method'] == method].copy()
        df_method = df_method.sort_values('Score', ascending=False).reset_index(drop=True)

        overlap_rates = []
        for k in top_ks:
            top_k = df_method.iloc[:k]
            tp_count = 0
            for _, row in top_k.iterrows():
                bait, prey = row['Bait'], row['Prey']
                if prey in go_cc_dict.get(bait, set()):
                    tp_count += 1
            rate = (tp_count / k) * 100
            overlap_rates.append(rate)

        plt.plot(top_ks, overlap_rates, label=method, color=colors.get(method, 'gray'), marker='o', markersize=3)

    # Plot individual features
    for feat in individual_features:
        df_feat = features_df[['Bait', 'Prey', feat]].copy()
        df_feat.columns = ['Bait', 'Prey', 'Score']
        df_feat['Method'] = feat

        df_feat_sorted = df_feat.sort_values('Score', ascending=False).reset_index(drop=True)

        overlap_rates = []
        for k in top_ks:
            top_k = df_feat_sorted.iloc[:k]
            tp_count = 0
            for _, row in top_k.iterrows():
                bait, prey = row['Bait'], row['Prey']
                if prey in go_cc_dict.get(bait, set()):
                    tp_count += 1
            rate = (tp_count / k) * 100
            overlap_rates.append(rate)

        plt.plot(top_ks, overlap_rates, label=feat, linestyle='dashed',
                color=feature_colors.get(feat, 'gray'))

    plt.xlabel('Number of Interactions (Top N)', fontsize=22)
    plt.ylabel('Database Overlap % (GO:CC)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(f'results/Benchmarking BioID DIA/overlap_vs_interactions_{gt_source}_with_features.pdf', dpi=300)
    plt.close()


