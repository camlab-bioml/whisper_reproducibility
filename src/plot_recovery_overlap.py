import os
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
    limma_df_raw,
    go_cc_df,
    gt_source='bg_large',
    outdir='results'
):
    """
    Plot recovery / database-overlap curves for WHISPER, SAINTq, SAINTexpress, limma,
    and the heuristic score, with optional individual feature baselines.

    Parameters
    ----------
    features_df : pd.DataFrame
        WHISPER feature/FDR output (must contain Bait, Prey, heuristic_score, predicted_probability).
    saintq_df_raw : pd.DataFrame
        Raw SAINTq table (Bait, PreyGene, BFDR, AvgP).
    saintexpress_df_raw : pd.DataFrame
        Raw SAINTexpress table (Bait, PreyGene, BFDR, AvgP).
    limma_df_raw : pd.DataFrame
        Raw limma table (Bait, Protein or Prey, adj.P.Val).
    go_cc_df : pd.DataFrame
        Ground-truth interactions (columns: Bait, Prey).
    gt_source : str
        Label for ground-truth type (e.g. 'go_large', 'bg_large') used in filenames.
    outdir : str
        Directory where plots are written (e.g. 'results/dataset1/Benchmarking BioID DIA').
    """

    os.makedirs(outdir, exist_ok=True)

    # Ground truth: restrict to identified preys
    identified_preys = set(features_df['Prey'].unique())
    go_cc_df = go_cc_df[go_cc_df['Prey'].isin(identified_preys)].copy()
    go_cc_dict = {
        bait: set(go_cc_df[go_cc_df['Bait'] == bait]['Prey'])
        for bait in features_df['Bait'].unique()
    }

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
                    'AvgP': [0.0] * len(missing_preys),
                })
                updated_df = pd.concat([updated_df, missing_df], ignore_index=True)
            updated_df = pd.concat([updated_df, df[df['Bait'] == bait]], ignore_index=True)
        return updated_df

    saintq_df = add_missing_preys(saintq_df_raw,        features_df, 'PreyGene')
    saintexpress_df = add_missing_preys(saintexpress_df_raw, features_df, 'PreyGene')

    # limma handling: ensure Prey col, add missing preys with adj.P.Val = 1.0 → limma_score = 1 - adj.P.Val
    limma_df_raw = limma_df_raw.rename(columns={'Protein': 'Prey'})  # harmless if already 'Prey'

    def add_missing_preys_limma(df, features_df):
        updated_df = pd.DataFrame()
        for bait in features_df['Bait'].unique():
            unique_preys = set(features_df[features_df['Bait'] == bait]['Prey'])
            existing_preys = set(df[df['Bait'] == bait]['Prey'])
            missing_preys = unique_preys - existing_preys
            if missing_preys:
                missing_df = pd.DataFrame({
                    'Bait': [bait] * len(missing_preys),
                    'Prey': list(missing_preys),
                    'adj.P.Val': [1.0] * len(missing_preys),
                })
                updated_df = pd.concat([updated_df, missing_df], ignore_index=True)
            updated_df = pd.concat([updated_df, df[df['Bait'] == bait]], ignore_index=True)
        return updated_df

    limma_df = add_missing_preys_limma(limma_df_raw, features_df)
    limma_df['adj.P.Val'] = limma_df['adj.P.Val'].astype(float).clip(0.0, 1.0)
    limma_df['limma_score'] = 1.0 - limma_df['adj.P.Val']

    # Method-specific tidy tables
    def prepare_method_df(df, bait_col, prey_col, score_col, method_name):
        df_clean = df[[bait_col, prey_col, score_col]].copy()
        df_clean.columns = ['Bait', 'Prey', 'Score']
        df_clean['Method'] = method_name
        return df_clean

    df_pu = features_df[['Bait', 'Prey', 'predicted_probability']].copy()
    df_pu.columns = ['Bait', 'Prey', 'Score']
    df_pu['Method'] = 'whisper'

    df_q  = prepare_method_df(saintq_df,       'Bait', 'PreyGene', 'AvgP',        'SAINTq')
    df_ex = prepare_method_df(saintexpress_df, 'Bait', 'PreyGene', 'AvgP',        'SAINTexpress')
    df_li = prepare_method_df(limma_df,        'Bait', 'Prey',     'limma_score', 'limma')

    # Normalize heuristic score globally
    features_df = features_df.copy()
    features_df['heuristic_score_norm'] = MinMaxScaler().fit_transform(
        features_df[['heuristic_score']]
    ).flatten()
    df_heuristic = features_df[['Bait', 'Prey', 'heuristic_score_norm']].copy()
    df_heuristic.columns = ['Bait', 'Prey', 'Score']
    df_heuristic['Method'] = 'heuristic score'

    # Combine all methods
    df_all = pd.concat([df_pu, df_q, df_ex, df_li, df_heuristic], ignore_index=True)

    # Evaluate overlap across top N interactions
    step = 10
    max_n = 2000
    top_ks = np.arange(step, max_n + step, step)

    colors = {
        'whisper':          '#1f77b4',
        'SAINTq':           '#ff7f0e',
        'SAINTexpress':     '#2ca02c',
        'limma':            '#9467bd',
        'heuristic score':  'crimson',
    }

    # --- Plot 1: main methods (no features) ---
    plt.figure(figsize=(8, 6))
    for method in df_all['Method'].unique():
        df_method = df_all[df_all['Method'] == method].copy()
        df_method = df_method.sort_values('Score', ascending=False).reset_index(drop=True)

        overlap_rates = []
        for k in top_ks:
            top_k = df_method.iloc[:k]
            tp_count = 0
            for _, row in top_k.iterrows():
                bait = row['Bait']
                prey = row['Prey']
                if prey in go_cc_dict.get(bait, set()):
                    tp_count += 1
            rate = (tp_count / k) * 100.0
            overlap_rates.append(rate)

        plt.plot(
            top_ks,
            overlap_rates,
            label=method,
            color=colors.get(method, 'gray'),
            marker='o',
            markersize=3,
        )

    plt.xlabel('Number of Interactions (Top N)', fontsize=22)
    plt.ylabel('Database Overlap % (GO:CC)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'overlap_vs_interactions_{gt_source}.pdf'),
                dpi=300)
    plt.close()

    # --- Plot 2: main methods + individual features ---
    individual_features = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff',
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
        'zero_or_neg_fc',
    ]

    feature_colors = {
        'log_fold_change':          'gray',
        'snr':                      'black',
        'mean_diff':                'darkblue',
        'median_diff':              'darkgreen',
        'replicate_fold_change_sd': 'darkred',
        'bait_cv':                  'darkorange',
        'bait_control_sd_ratio':    'darkviolet',
        'zero_or_neg_fc':           'brown',
    }

    plt.figure(figsize=(8, 6))

    # Main methods
    for method in df_all['Method'].unique():
        df_method = df_all[df_all['Method'] == method].copy()
        df_method = df_method.sort_values('Score', ascending=False).reset_index(drop=True)

        overlap_rates = []
        for k in top_ks:
            top_k = df_method.iloc[:k]
            tp_count = 0
            for _, row in top_k.iterrows():
                bait = row['Bait']
                prey = row['Prey']
                if prey in go_cc_dict.get(bait, set()):
                    tp_count += 1
            rate = (tp_count / k) * 100.0
            overlap_rates.append(rate)

        plt.plot(
            top_ks,
            overlap_rates,
            label=method,
            color=colors.get(method, 'gray'),
            marker='o',
            markersize=3,
        )

    # Individual features
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
                bait = row['Bait']
                prey = row['Prey']
                if prey in go_cc_dict.get(bait, set()):
                    tp_count += 1
            rate = (tp_count / k) * 100.0
            overlap_rates.append(rate)

        plt.plot(
            top_ks,
            overlap_rates,
            label=feat,
            linestyle='dashed',
            color=feature_colors.get(feat, 'gray'),
        )

    plt.xlabel('Number of Interactions (Top N)', fontsize=22)
    plt.ylabel('Database Overlap % (GO:CC)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'overlap_vs_interactions_{gt_source}_with_features.pdf'),
                dpi=300)
    plt.close()