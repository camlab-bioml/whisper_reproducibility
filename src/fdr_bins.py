"""True positives and precision in FDR bins (Fig. 5, Supplementary Fig. 14c)."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.benchmark import (FEATURE_NAMES, add_missing_preys, is_any_gene_in_gold,
                           load_uniprot_to_gene, make_to_gene)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

FDR_BINS = [0.0, 0.01, 0.05]
BIN_LABELS = ['0-0.01', '0.01-0.05']
COLORS = {'PU Learning': '#003f5c', 'WHISPER': '#003f5c', 'SAINTq': '#ffb347',
          'SAINTexpress': '#ff7f0e', 'Composite': '#7a5195',
          'Composite score': '#7a5195', 'limma': '#d62728'}
COLORS.update(dict(zip(FEATURE_NAMES, sns.color_palette('YlGnBu', n_colors=len(FEATURE_NAMES)))))


def compute_precision_and_tp(df, fdr_column, bins=FDR_BINS):
    tmp = df.copy()
    tmp['FDR_bin'] = pd.cut(tmp[fdr_column], bins=bins, labels=BIN_LABELS,
                            include_lowest=True)
    results = tmp.groupby('FDR_bin').apply(
        lambda x: (x['true_label'].sum(), len(x),
                   x['true_label'].sum() / len(x) if len(x) > 0 else 0))
    return results.apply(lambda x: {'TP': x[0], 'Total': x[1], 'Precision': x[2]}).to_dict()


def run_dataset(cfg):
    """cfg: features, saintq, saintexpress, limma (paths or None), gts
    [(source, file)], outdir, fdr_bins_map_ids, gaf_file."""
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    features_df = pd.read_csv(cfg['features'])

    others = {}   # method -> (table, prey column, FDR column)
    if cfg.get('saintq'):
        q = pd.read_csv(cfg['saintq'], sep='\t').rename(columns={'Prey': 'PreyGene'})
        others['SAINTq'] = (add_missing_preys(q, features_df, 'PreyGene'), 'PreyGene', 'BFDR')
    if cfg.get('saintexpress'):
        e = pd.read_csv(cfg['saintexpress'], sep='\t')
        others['SAINTexpress'] = (add_missing_preys(e, features_df, 'PreyGene'), 'PreyGene', 'BFDR')
    if cfg.get('limma'):
        lm = pd.read_csv(cfg['limma'], sep='\t').rename(columns={'Protein': 'Prey'})
        others['limma'] = (add_missing_preys(lm, features_df, 'Prey'), 'Prey',
                           'adj.P.Val.one_sided')

    if cfg.get('fdr_bins_map_ids'):
        # padded first, then mapped, as in the research blocks
        to_gene = make_to_gene(load_uniprot_to_gene(cfg['gaf_file']))
        features_df['Prey'] = features_df['Prey'].astype(str).map(to_gene)
        for table, col, _f in others.values():
            table[col] = table[col].astype(str).map(to_gene)

    methods = ['PU Learning'] + [m for m in ('SAINTq', 'SAINTexpress', 'limma') if m in others]
    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(set(features_df['Prey'].unique()))]
        gold_by_bait = {b: set(gt[gt['Bait'] == b]['Prey']) for b in features_df['Bait'].unique()}

        rows = []
        tp = {m: {b: 0 for b in BIN_LABELS} for m in methods}
        total = {m: {b: 0 for b in BIN_LABELS} for m in methods}
        for bait in features_df['Bait'].unique():
            gold = gold_by_bait.get(bait, set())
            per_method = {}
            w = features_df[features_df['Bait'] == bait].copy()
            w['true_label'] = w['Prey'].apply(lambda x: is_any_gene_in_gold(x, gold))
            per_method['PU Learning'] = compute_precision_and_tp(w, 'FDR')
            for m, (table, col, fdr_col) in others.items():
                d = table[table['Bait'] == bait].copy()
                d['true_label'] = d[col].apply(lambda x: is_any_gene_in_gold(x, gold))
                per_method[m] = compute_precision_and_tp(d, fdr_col)
            for b in BIN_LABELS:
                for m in methods:
                    r = per_method[m][b]
                    rows.append({'Bait': bait, 'FDR Bin': b, 'Method': m,
                                 'Precision': r['Precision'], 'TP': r['TP']})
                    tp[m][b] += r['TP']
                    total[m][b] += r['Total']
        pd.DataFrame(rows).to_csv(f'{outdir}/precision_tp_results_by_bin_{gt_source}.csv',
                                  index=False)

        plot_df = pd.DataFrame([{'Method': m, 'FDR Bin': b,
                                 'Precision': tp[m][b] / total[m][b] if total[m][b] > 0 else 0,
                                 'TP': tp[m][b]} for m in methods for b in BIN_LABELS])
        out = plot_df.copy()
        out['Total'] = [total[r['Method']][r['FDR Bin']] for _, r in plot_df.iterrows()]
        out['ground_truth'] = gt_source
        out.to_csv(f'{outdir}/tp_precision_source_data_{gt_source}.csv', index=False)
        _plot(plot_df, f'{outdir}/tp_precision_barplot_{gt_source}.pdf')


def _plot(plot_df, out_pdf):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 14), sharex=True)
    for ax, y, label in ((axes[0], 'TP', 'True Positives'), (axes[1], 'Precision', 'Precision')):
        sns.barplot(data=plot_df, x='FDR Bin', y=y, hue='Method', palette=COLORS, ax=ax,
                    order=BIN_LABELS)
        ax.set_ylabel(label, fontsize=26)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        if ax.legend_:
            ax.legend_.remove()
    axes[0].set_xlabel('', fontsize=26)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('FDR Bin', fontsize=26)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Method', ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), fontsize=18, title_fontsize=20)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_pdf, dpi=300)
    plt.close()
