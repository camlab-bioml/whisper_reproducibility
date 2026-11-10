"""AUC at ranking depths 300, 200, 100 and 50, and paired differences
(Supplementary Figs. 3 and 4)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.benchmark import (ALL_METHODS, FEATURE_NAMES, N_BOOTSTRAP, OLD_METHODS,
                           is_any_gene_in_gold, load_new_methods, load_old_methods,
                           paired_bootstrap_aucs, paired_bootstrap_aucs_by_bait,
                           paired_difference_ci, tie_averaged_labels)

TOP_NS = [50, 100, 200, 300]
DEPTH_LABEL = {n: f'Top {n}' for n in TOP_NS}
DEPTH_ORDER = [DEPTH_LABEL[n] for n in TOP_NS]


PRECISION_NS = [10, 25, 50, 100, 200, 300]

COMPARATORS = ['Heuristic score', 'SAINTq', 'SAINTexpress', 'limma']


LABEL = {'Composite score': 'Heuristic score'}

METHOD_COLORS_RANGE = {
    'WHISPER': '#003f5c',
    'Heuristic score': '#7a5195',
    'SAINTq': '#ffb347',
    'SAINTexpress': '#ff7f0e',
    'limma': '#d62728',
}
METHOD_ORDER = ['WHISPER', 'Heuristic score', 'SAINTq', 'SAINTexpress', 'limma']

UNIT_STYLE = {
    'pair': dict(color='#2a78d6', marker='o', label='pairs (optimistic)'),
    'bait': dict(color='#eb6834', marker='s', label='baits (conservative)'),
}
GRID_KW = dict(color='#d8d8d4', linewidth=0.6, zorder=0)
GT_YLABEL = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}

def build_specs(cfg, gt_file):
    """Every method and feature labelled under one ground truth, in the order
    plot_precision_overall_rank_all.py's run_dataset builds them, so the
    shared pair alignment and the bootstrap draws are its own."""
    features_df, frames = load_new_methods(cfg, None)
    frames.update(load_old_methods(cfg, features_df))
    frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')

    baits = list(features_df['Bait'].unique())
    identified = set(features_df['Prey'].astype(str))
    present = [m for m in ALL_METHODS if m in frames]

    gt = pd.read_csv(gt_file)
    gt = gt[gt['Prey'].isin(identified)]
    gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}

    specs = []
    for m in present:
        df, prey_col, score_col = frames[m]
        d = df.copy()
        d['true_label'] = [is_any_gene_in_gold(p, gold.get(b, set()))
                           for b, p in zip(d['Bait'], d[prey_col])]
        specs.append((m, d, prey_col, score_col))
    whisper_df = specs[[s[0] for s in specs].index('WHISPER')][1]
    for feat in [f for f in FEATURE_NAMES if f in whisper_df.columns]:
        specs.append((feat, whisper_df, 'Prey', feat))
    return specs, baits


def aligned_labels(specs):

    keys = [list(zip(df['Bait'].astype(str), df[prey_col].astype(str)))
            for _n, df, prey_col, _sc in specs]
    common = set(keys[0])
    for k in keys[1:]:
        common &= set(k)
    seen, order = set(), []
    for k in keys[0]:
        if k in common and k not in seen:
            seen.add(k)
            order.append(k)
    out = {}
    for (name, df, prey_col, score_col), k in zip(specs, keys):
        d = df.copy()
        d['_pair_key'] = k
        d = d.drop_duplicates('_pair_key').set_index('_pair_key').reindex(order)
        score = pd.to_numeric(d[score_col], errors='coerce').to_numpy(dtype=float)
        out[name] = tie_averaged_labels(
            np.where(np.isnan(score), -np.inf, score),
            d['true_label'].to_numpy(dtype=float))
    return out


def measure(cfg, gt_source, gt_file):

    specs, baits = build_specs(cfg, gt_file)
    depths = {DEPTH_LABEL[n]: n for n in TOP_NS}
    depths['Total AUC'] = None
    boots = {'pair': paired_bootstrap_aucs(specs, depths,
                                           n_bootstrap=N_BOOTSTRAP, seed=0),
             'bait': paired_bootstrap_aucs_by_bait(specs, depths, baits,
                                                   n_bootstrap=N_BOOTSTRAP,
                                                   seed=0)}
    ranked = aligned_labels(specs)
    shown = [m for m in OLD_METHODS if m in boots['pair']]

    rows = []
    for m in shown:
        label = LABEL.get(m, m)
        for lab, topn in depths.items():
            b = boots['pair'][m][lab]
            rows.append(dict(dataset=cfg['name'], ground_truth=gt_source,
                             Method=label, Depth=lab,
                             topn=(topn or len(ranked[m])),
                             AUC=float(b.mean()), AUC_std=float(b.std())))
        for d in PRECISION_NS:
            t = ranked[m][:min(d, len(ranked[m]))]
            rows.append(dict(dataset=cfg['name'], ground_truth=gt_source,
                             Method=label, Depth=f'precision@{d}', topn=d,
                             AUC=np.nan,
                             precision=float(t.mean()) if len(t) else np.nan))
    point = pd.DataFrame(rows)

    diffs = []
    for unit, boot in boots.items():
        for other in [m for m in shown if m != 'WHISPER']:
            for lab in depths:
                r = paired_difference_ci(boot, 'WHISPER', other, lab)
                diffs.append(dict(dataset=cfg['name'], ground_truth=gt_source,
                                  reference=LABEL.get(other, other), depth=lab,
                                  topn=depths[lab], resampling_unit=unit,
                                  mean_difference=r['mean_difference'],
                                  ci_low=r['ci_low'], ci_high=r['ci_high'],
                                  frac_positive=r['frac_replicates_positive']))
    diff = pd.DataFrame(diffs)
    diff['excludes_0'] = ~((diff['ci_low'] <= 0) & (0 <= diff['ci_high']))
    return point, diff


# =============================================================================
# Figures
# =============================================================================

# Deepest cut-off on the left: Top 300, 200, 100, 50.
PLOT_ORDER = DEPTH_ORDER[::-1]

def plot_bars(point, cfg, gt_source, out_pdf):

    a = point[point['Depth'].isin(PLOT_ORDER) & point['AUC'].notna()]
    methods = [m for m in METHOD_ORDER if m in set(a['Method'])]

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    sns.barplot(data=a, x='Depth', y='AUC', hue='Method', order=PLOT_ORDER,
                hue_order=methods, palette=METHOD_COLORS_RANGE, ax=ax, errorbar=None)
    # Error bars: SD over the pair bootstrap, each placed on its own bar.
    width = 0.8 / len(methods)
    for i, depth in enumerate(PLOT_ORDER):
        for j, m in enumerate(methods):
            r = a[(a['Depth'] == depth) & (a['Method'] == m)]
            if r.empty:
                continue
            ax.errorbar(i - 0.4 + width * (j + 0.5), r['AUC'].iloc[0],
                        yerr=r['AUC_std'].iloc[0], fmt='none', ecolor='black',
                        elinewidth=1, capsize=3)
    ax.set_ylabel(f'AUCPR - {GT_YLABEL[gt_source]}', fontsize=21)
    ax.set_xlabel('')
    ax.tick_params(labelsize=18)
    ax.set_ylim(0, 1)          # fixed, so panels align across datasets
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(title='Method', ncol=min(len(methods), 5), loc='lower center',
              bbox_to_anchor=(0.5, -0.26), fontsize=11, title_fontsize=11,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_paired(diff, cfg, gt_source, out_pdf):

    refs = [c for c in COMPARATORS if c in set(diff['reference'])]
    if not refs:
        return False
    order = PLOT_ORDER
    x0 = np.arange(len(order))
    span = 0.6                       # width the comparisons spread over per depth
    offsets = (np.linspace(-span / 2, span / 2, len(refs))
               if len(refs) > 1 else np.zeros(1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (unit, st) in zip(axes, UNIT_STYLE.items()):
        ax.axhline(0, linestyle=(0, (4, 3)), linewidth=1.2, color='#8a8a85',
                   zorder=1)
        for off, ref in zip(offsets, refs):
            s_ = (diff[(diff['reference'] == ref) & (diff['resampling_unit'] == unit)]
                  .set_index('depth').reindex(order))
            x = x0 + off
            y = s_['mean_difference'].to_numpy()
            lo = (s_['mean_difference'] - s_['ci_low']).to_numpy()
            hi = (s_['ci_high'] - s_['mean_difference']).to_numpy()
            color = METHOD_COLORS_RANGE[ref]
            ax.errorbar(x, y, yerr=[lo, hi], fmt='o', color=color,
                        markersize=6.5, linewidth=0, elinewidth=1.6,
                        capsize=3, markeredgecolor='#fcfcfb',
                        markeredgewidth=1.2, zorder=3, label=f'vs {ref}')
            # Hollow centre where the interval crosses zero, so the distinction
            # survives greyscale printing.
            for xi, (_i, r) in zip(x, s_.iterrows()):
                if pd.isna(r['ci_low']) or (r['ci_low'] <= 0 <= r['ci_high']):
                    ax.plot([xi], [r['mean_difference']], marker='o',
                            color='#fcfcfb', markersize=3.6, zorder=4)
        ax.set_xticks(x0)
        ax.set_xticklabels([o.replace('Top ', '') for o in order], fontsize=15)
        ax.set_xlabel('ranking depth', fontsize=16.5)
        ax.set_title({'pair': 'Pairs resampling', 'bait': 'Baits resampling'}[unit],
                     fontsize=11.5)
        ax.grid(axis='y', **GRID_KW)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=15)
    axes[0].set_ylabel(f'AUCPR - {GT_YLABEL[gt_source]} difference\n'
                       f'(WHISPER − other)', fontsize=16.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(refs), frameon=False, fontsize=10,
               loc='lower center', bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    return True

def run_dataset(cfg):
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    for gt_source, gt_file in cfg['gts']:
        point, diff = measure(cfg, gt_source, gt_file)
        point.to_csv(os.path.join(outdir, f'auc_range_source_data_{gt_source}.csv'), index=False)
        diff.to_csv(os.path.join(outdir, f'auc_range_paired_{gt_source}.csv'), index=False)
        plot_bars(point, cfg, gt_source,
                  os.path.join(outdir, f'auc_range_{gt_source}_main_methods.pdf'))
        plot_paired(diff, cfg, gt_source, os.path.join(outdir, f'auc_range_paired_{gt_source}.pdf'))
