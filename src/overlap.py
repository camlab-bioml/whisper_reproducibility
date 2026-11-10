"""Overlap of top-ranked interactions with the references (Fig. 4,
Supplementary Fig. 11)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import (FEATURE_NAMES, METHOD_COLORS, is_any_gene_in_gold,
                           load_new_methods, load_old_methods,
                           tie_averaged_labels)

STEP = 10
MAX_N = 2000
GT_LABEL = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}
WHISPER_LABEL = 'PU Learning'
ALL_METHODS = [WHISPER_LABEL, 'Composite score', 'SAINTq', 'SAINTexpress',
               'limma', 'CompPASS', 'SFINX', 'PP-NSAF', 'MiST']
MAIN_METHODS = [WHISPER_LABEL, 'Composite score', 'SAINTq', 'SAINTexpress',
                'limma']
HEURISTIC = 'Composite score'
PALETTE = dict(METHOD_COLORS)
PALETTE[WHISPER_LABEL] = METHOD_COLORS['WHISPER']


def overlap_curve(scores, hits, top_ks):
    cum = np.cumsum(tie_averaged_labels(scores, hits))
    ks = np.minimum(top_ks, len(cum))
    return 100.0 * cum[ks - 1] / ks


def draw(curves, out_pdf, top_ks, ylabel, features=None):
    plt.figure(figsize=(8, 6))
    for label, rates in curves:
        plt.plot(top_ks, rates, label=label,
                 color=PALETTE.get(label, 'gray'), marker='o', markersize=3)
    for label, rates in (features or []):
        plt.plot(top_ks, rates, label=label, linestyle='dashed',
                 color=PALETTE.get(label, 'gray'))
    plt.xlabel('Number of Interactions', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.ylim(0, 100)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def run_dataset(cfg):
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)

    features_df, frames = load_new_methods(cfg, None)
    frames.update(load_old_methods(cfg, features_df))
    frames[WHISPER_LABEL] = (features_df, 'Prey', 'predicted_probability')
    frames.pop('WHISPER', None)
    present = [m for m in ALL_METHODS if m in frames]

    parts = []
    for m in present:
        df, prey_col, score_col = frames[m]
        parts.append(df[['Bait', prey_col, score_col]]
                     .set_axis(['Bait', 'Prey', 'Score'], axis=1).assign(Method=m))
    df_all = pd.concat(parts, ignore_index=True)

    top_ks = np.arange(STEP, min(MAX_N, len(features_df)) + STEP, STEP)
    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(set(features_df['Prey'].astype(str)))]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in features_df['Bait'].unique()}
        ylabel = f'Database Overlap % ({GT_LABEL[gt_source]})'

        curves = []
        for m in present:
            d = df_all[df_all['Method'] == m]
            hits = [is_any_gene_in_gold(p, gold.get(b, set()))
                    for b, p in zip(d['Bait'], d['Prey'].astype(str))]
            curves.append((m, overlap_curve(d['Score'], hits, top_ks)))
        fhits = [is_any_gene_in_gold(p, gold.get(b, set()))
                 for b, p in zip(features_df['Bait'], features_df['Prey'].astype(str))]
        features = [(f, overlap_curve(features_df[f], fhits, top_ks))
                    for f in FEATURE_NAMES if f in features_df.columns]

        main = [c for c in curves
                if cfg.get('all_methods_in_main') or c[0] in MAIN_METHODS]
        variants = {
            'main': main,
            'main_no_heuristic': [c for c in main if c[0] != HEURISTIC],
            'all_methods': curves,
            'all_methods_no_heuristic': [c for c in curves if c[0] != HEURISTIC],
            'whisper_vs_heuristic': [c for c in curves if c[0] in (WHISPER_LABEL, HEURISTIC)],
        }
        for suffix, sel in variants.items():
            draw(sel, f'{outdir}/overlap_vs_interactions_all_{gt_source}_{suffix}.pdf',
                 top_ks, ylabel)
        draw(curves, f'{outdir}/overlap_vs_interactions_all_{gt_source}_with_features.pdf',
             top_ks, ylabel, features=features)

        rows = [pd.DataFrame({'Method': label, 'TopN': top_ks, 'Overlap_pct': rates,
                              'ground_truth': gt_source})
                for label, rates in curves + features]
        pd.concat(rows, ignore_index=True).to_csv(
            f'{outdir}/overlap_source_data_all_{gt_source}.csv', index=False)
