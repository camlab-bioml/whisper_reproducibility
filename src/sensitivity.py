"""Sensitivity to the number of weak-label positives and negatives
(Supplementary Figs. 18 and 19)."""

import io
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import benchmark as pr
from src.feature_engineering import FEATURE_COLUMNS
from src.train_and_fdr import build_model, monotone_q, strong_baits, weak_labels

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

N_POS_LIST = [5, 10, 15, 20, 25, 30]
N_NEG_LIST = [100, 200, 500, 1000]
DIAG_POS = [1, 5, 10, 15, 20, 25, 30]
DIAG_NEG = [100, 200, 500, 1000]
N_DECOY_REPLICATES = 30
GT_LABEL = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}
FILE_TAG = {'go_large': 'go', 'bg_large': 'bg'}
METRICS = [
    ('fdr_hits', 'Hits Below 1% FDR'),
    ('kl_divergence', 'KL Divergence (Real vs Decoy)'),
    ('wasserstein_distance', 'Wasserstein Distance'),
    ('roc_auc', 'ROC AUC (Real vs Decoy)'),
    ('median_score_difference', 'Median Score Difference (Real - Decoy)'),
]


def _fit(df_real, n_pos, n_neg):
    np.random.seed(42)
    df_elig = df_real[~df_real['is_inert']]
    x_real = df_real[FEATURE_COLUMNS].values
    y = weak_labels(df_real, df_elig, set(strong_baits(df_real, df_elig)), n_pos, n_neg)
    labeled = y[y != 0].index
    scaler = StandardScaler()
    model = build_model()
    model.fit(scaler.fit_transform(x_real[labeled]), y.loc[labeled])
    return model, scaler, model.predict_proba(scaler.transform(x_real))[:, 1]


def _as_saved(probs):

    buf = pd.DataFrame({'p': probs}).to_csv(index=False)
    return pd.read_csv(io.StringIO(buf))['p'].to_numpy()


def _barplot(table, value, std, ylabel, out_pdf):
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=table, x='Positives', y=value, hue='Negatives', palette='Blues',
                     errorbar=None, order=N_POS_LIST, hue_order=N_NEG_LIST)
    width = 0.8 / len(N_NEG_LIST)
    for _, row in table.iterrows():
        x = N_POS_LIST.index(row['Positives']) - 0.4 + width * (N_NEG_LIST.index(row['Negatives']) + 0.5)
        ax.errorbar(x, row[value], yerr=row[std], fmt='none', ecolor='black', capsize=3)
    plt.xlabel('Number of Positives', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.tick_params(axis='both', labelsize=23)
    plt.ylim(0, 1)
    plt.legend(title='Negatives', fontsize=12, title_fontsize=14, loc=(1.05, 1))
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def posneg_auc(cfg):
    out_dir = cfg['outdir']
    df = pd.read_csv(cfg['features']).sort_values(['Bait', 'Prey']).reset_index(drop=True)
    runs = {(p, n): _as_saved(_fit(df, p, n)[2]) for p in N_POS_LIST for n in N_NEG_LIST}

    scored = df[['Bait', 'Prey']].copy()
    if cfg['gaf']:
        scored['Prey'] = scored['Prey'].astype(str).map(
            pr.make_to_gene(pr.load_uniprot_to_gene(cfg['gaf_file'])))
    for (p, n), probs in runs.items():
        scored[f'p_{p}_{n}'] = probs
    baits = list(scored['Bait'].unique())
    identified = set(scored['Prey'].astype(str))

    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(identified)]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}
        d = scored.copy()
        d['true_label'] = [pr.is_any_gene_in_gold(p, gold.get(b, set()))
                           for b, p in zip(d['Bait'], d['Prey'])]
        boot = pr.paired_bootstrap_aucs([(k, d, 'Prey', f'p_{k[0]}_{k[1]}') for k in runs],
                                        pr.DEPTHS, seed=0)
        table = pd.DataFrame([{'Positives': k[0], 'Negatives': k[1],
                               'Mean AUC': boot[k]['Total AUC'].mean(),
                               'Std AUC': boot[k]['Total AUC'].std(),
                               'Mean AUC@Top300': boot[k]['AUC@Top300'].mean(),
                               'Std AUC@Top300': boot[k]['AUC@Top300'].std()} for k in runs])
        tag = FILE_TAG[gt_source]
        table.to_csv(os.path.join(out_dir, f'puppi_pr_results_by_pos_neg_{tag}.csv'), index=False)
        ylabel = f'AUCPR - {GT_LABEL[gt_source]}'
        _barplot(table, 'Mean AUC', 'Std AUC', ylabel,
                 os.path.join(out_dir, f'puppi_auc_barplot_by_pos_neg_{tag}.pdf'))
        _barplot(table, 'Mean AUC@Top300', 'Std AUC@Top300', ylabel,
                 os.path.join(out_dir, f'puppi_auc_barplot_by_pos_neg_top300_{tag}.pdf'))


def _diagnostics(df_real, n_pos, n_neg):
    model, scaler, all_probs = _fit(df_real, n_pos, n_neg)
    df_elig = df_real[~df_real['is_inert']]
    decoys = []
    for rep in range(N_DECOY_REPLICATES):
        for i, bait in enumerate(df_elig['Bait'].unique()):
            np.random.seed(42 + rep * 1000 + i)
            d = df_elig[df_elig['Bait'] == bait].copy()
            for col in FEATURE_COLUMNS:
                d[col] = np.random.permutation(d[col].values)
            decoys.extend(model.predict_proba(scaler.transform(d[FEATURE_COLUMNS].values))[:, 1])
    decoys = np.array(decoys)
    real = all_probs[~df_real['is_inert'].to_numpy()]
    fdr = monotone_q(real, decoys, N_DECOY_REPLICATES, all_probs)
    h_real, _ = np.histogram(real, bins=100, range=(0, 1), density=True)
    h_decoy, _ = np.histogram(decoys, bins=100, range=(0, 1), density=True)
    return {'initial_positives': n_pos, 'initial_negatives': n_neg,
            'fdr_hits': int((fdr <= 0.01).sum()),
            'kl_divergence': entropy(h_real + 1e-10, h_decoy + 1e-10),
            'wasserstein_distance': wasserstein_distance(real, decoys),
            'roc_auc': roc_auc_score(np.concatenate([np.ones_like(real), np.zeros_like(decoys)]),
                                     np.concatenate([real, decoys])),
            'median_score_difference': np.median(real) - np.median(decoys)}


def label_diagnostics(features_csv, out_dir):
    df = pd.read_csv(features_csv).sort_values(['Bait', 'Prey']).reset_index(drop=True)
    table = pd.DataFrame([_diagnostics(df, p, n) for p in DIAG_POS for n in DIAG_NEG])
    table.to_csv(os.path.join(out_dir, 'pu_learning_optimization.csv'), index=False)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    for ax, (metric, title) in zip(axes, METRICS):
        sns.lineplot(data=table, x='initial_positives', y=metric, hue='initial_negatives',
                     marker='o', ax=ax, legend=False)
        ax.set_xlabel('Initial Positives', fontsize=16)
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=16)
        ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pu_learning_optimization_plot.pdf'), dpi=300)
    plt.close()
