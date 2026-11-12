"""Transfer across datasets (Supplementary Fig. 17)."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import benchmark as pr
from src.feature_engineering import FEATURE_COLUMNS
from src.train_and_fdr import fit

OWN = 'WHISPER'
TRANSFER = 'WHISPER (trained on BioID dataset 1)'
pr.METHOD_COLORS[TRANSFER] = '#eb6834'
GT_LABELS = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}
STYLE = {OWN: dict(color='#2a78d6', marker='o', label='trained on the dataset itself'),
         TRANSFER: dict(color='#eb6834', marker='s', label='trained on BioID dataset 1')}


def _score_target(cfg, scaler, model, out_root, summary):
    own = pd.read_csv(cfg['features'])
    if cfg['gaf']:
        own['Prey'] = own['Prey'].astype(str).map(
            pr.make_to_gene(pr.load_uniprot_to_gene(cfg['gaf_file'])))
    out_dir = os.path.join(out_root, cfg['name'])
    os.makedirs(out_dir, exist_ok=True)
    transfer = own.copy()
    transfer['predicted_probability'] = model.predict_proba(
        scaler.transform(own[FEATURE_COLUMNS].values))[:, 1]
    baits = list(own['Bait'].unique())
    identified = set(own['Prey'].astype(str))
    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(identified)]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}
        specs = []
        for name, df in ((OWN, own), (TRANSFER, transfer)):
            d = df.copy()
            d['true_label'] = [pr.is_any_gene_in_gold(p, gold.get(b, set()))
                               for b, p in zip(d['Bait'], d['Prey'])]
            specs.append((name, d, 'Prey', 'predicted_probability'))
        boot = pr.paired_bootstrap_aucs(specs, pr.DEPTHS, seed=0)
        auc_df = pd.DataFrame([{'Method': n,
                                'Total AUC': float(boot[n]['Total AUC'].mean()),
                                'Total AUC Std': float(boot[n]['Total AUC'].std()),
                                'AUC@Top300': float(boot[n]['AUC@Top300'].mean()),
                                'AUC@Top300 Std': float(boot[n]['AUC@Top300'].std())}
                               for n, _d, _p, _s in specs])
        pr.auc_bar(auc_df, os.path.join(out_dir, f'auc_barplot_{gt_source}.pdf'), 2,
                   (0.5, -0.32), title=cfg['name'])
        out = auc_df.copy()
        for lab in pr.DEPTHS:
            c = pr.paired_difference_ci(boot, TRANSFER, OWN, lab)
            out[f'{lab} diff'] = c['mean_difference']
            out[f'{lab} diff CI low'] = c['ci_low']
            out[f'{lab} diff CI high'] = c['ci_high']
            summary.append({'dataset': cfg['name'], 'ground_truth': gt_source, 'depth': lab,
                            'own': float(boot[OWN][lab].mean()),
                            'transferred': float(boot[TRANSFER][lab].mean()),
                            'difference': c['mean_difference'],
                            'ci_low': c['ci_low'], 'ci_high': c['ci_high']})
        out['ground_truth'] = gt_source
        out.to_csv(os.path.join(out_dir, f'auc_source_data_{gt_source}.csv'), index=False)


def run(source_features_csv, source_positives, targets, out_root):
    os.makedirs(out_root, exist_ok=True)
    _df, model, scaler, _y = fit(pd.read_csv(source_features_csv), source_positives, 200)
    summary = []
    for cfg in targets:
        _score_target(cfg, scaler, model, out_root, summary)
    pd.DataFrame(summary).to_csv(os.path.join(out_root, 'transfer_summary.csv'), index=False)
    _summary_plot([c['name'] for c in targets], out_root)


def _summary_plot(names, out_root):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    metrics = ['Total AUC', 'AUC@Top300']
    n = len(names)
    fig, axes = plt.subplots(len(GT_LABELS), len(metrics),
                             figsize=(1.35 * n * len(metrics) + 1.0, 3.7 * len(GT_LABELS)),
                             squeeze=False, sharex=True)
    x = np.arange(n, dtype=float)
    for i, (gt, gt_label) in enumerate(GT_LABELS.items()):
        tabs = [pd.read_csv(os.path.join(out_root, ds, f'auc_source_data_{gt}.csv'))
                .set_index('Method') for ds in names]
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for name, dx in ((OWN, -0.16), (TRANSFER, 0.16)):
                st = STYLE[name]
                ax.bar(x + dx, [t.loc[name, metric] for t in tabs], width=0.32,
                       yerr=[t.loc[name, f'{metric} Std'] for t in tabs], color=st['color'],
                       capsize=3, error_kw=dict(elinewidth=1.2, ecolor='#333333'), zorder=3,
                       label=st['label'] if (i == 0 and j == 0) else None)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=35, ha='right', fontsize=13.5)
            ax.set_xlim(-0.6, n - 0.4)
            for k in range(n - 1):
                ax.axvline(k + 0.5, color='#d0d0cc', linewidth=0.8, zorder=0)
            ax.set_ylabel(metric, fontsize=16.5)
            ax.set_ylim(0, 1)
            ax.set_title(f'{gt_label} — {metric}', fontsize=11.5, loc='left')
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', which='both', labelsize=14.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('WHISPER: trained on each dataset against trained on BioID dataset 1', fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(out_root, 'transfer_summary.pdf'), dpi=300)
    plt.close(fig)
