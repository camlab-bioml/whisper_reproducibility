"""Hyperparameter tuning on the weak labels (Supplementary Fig. 16)."""

import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_COLUMNS
from src.train_and_fdr import strong_baits, weak_labels

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

GRID = [dict(max_features=f, min_samples_leaf=m, class_weight=c, n_estimators=n)
        for f, m, c, n in itertools.product(['sqrt', 0.5, None], [1, 3, 10],
                                            [None, 'balanced'], [100, 50, 200])]
BASE_PARAMS = dict(n_estimators=100, max_features='sqrt', min_samples_leaf=1,
                   class_weight=None, criterion='gini')
SEED = 42
BAGGING_N = 100
GROUND_TRUTHS = ['go_large', 'bg_large']
GT_LABEL = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}
DEPTHS = ['Total AUC', 'AUC@Top300']
STYLE = {'vanilla': dict(color='#2a78d6', marker='o', label='defaults'),
         'bestconfig': dict(color='#eb6834', marker='s', label='CV-selected config')}


def config_label(cfg):
    f = {'sqrt': 'sqrt', None: 'all'}.get(cfg['max_features'], cfg['max_features'])
    w = 'bal' if cfg['class_weight'] else '-'
    return f"{f}/{cfg['min_samples_leaf']}/{w}/{cfg['n_estimators']}"


DEPLOYED_LABEL = config_label(GRID[0])
CONFIG_BY_LABEL = {config_label(c): c for c in GRID}


def _fit(x, y, labeled, cfg, oob=False):
    scaler = StandardScaler()
    rf = RandomForestClassifier(random_state=SEED, **{**BASE_PARAMS, **cfg})
    model = BaggingClassifier(estimator=rf, n_estimators=BAGGING_N, random_state=SEED,
                              oob_score=oob)
    model.fit(scaler.fit_transform(x[labeled]), y.loc[labeled])
    return model, scaler


def _internal_row(x, y, labeled, strat, model, cfg, n_splits=5):
    yl = y.loc[labeled].to_numpy()
    pos = (yl == 1).astype(int)
    k = min(n_splits, int(pd.Series(strat).value_counts().min()))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs, losses = [], []
    for tr, te in skf.split(np.zeros(len(yl)), strat):
        sc = StandardScaler()
        m, _ = _fit(x, y, labeled[tr], cfg)
        p = m.predict_proba(sc.fit(x[labeled[tr]]).transform(x[labeled[te]]))[
            :, list(m.classes_).index(1)]
        aucs.append(roc_auc_score(pos[te], p))
        losses.append(log_loss(pos[te], p, labels=[0, 1]))
    oob = model.oob_decision_function_[:, list(model.classes_).index(1)]
    ok = np.isfinite(oob)
    return {'config': config_label(cfg), 'n_labelled': int(len(labeled)),
            'n_positives': int(pos.sum()), 'n_splits': k,
            'cv_auc': float(np.mean(aucs)), 'cv_auc_sd': float(np.std(aucs)),
            'cv_logloss': float(np.mean(losses)), 'cv_logloss_sd': float(np.std(losses)),
            'oob_auc': float(roc_auc_score(pos[ok], oob[ok])),
            'oob_error': float(np.mean((oob[ok] > 0.5) != (pos[ok] == 1))),
            'oob_logloss': float(log_loss(pos[ok], oob[ok], labels=[0, 1])),
            'oob_pos_acc': float(np.mean(oob[ok][pos[ok] == 1] > 0.5)),
            'oob_neg_acc': float(np.mean(oob[ok][pos[ok] == 0] <= 0.5))}


def sweep(features_csv, initial_positives, out_csv, initial_negatives=200):
    df = pd.read_csv(features_csv).sort_values(['Bait', 'Prey']).reset_index(drop=True)
    eligible = df[~df['is_inert']]
    y = weak_labels(df, eligible, strong_baits(df, eligible), initial_positives,
                    initial_negatives)
    labeled = y[y != 0].index
    x = df[FEATURE_COLUMNS].to_numpy(float)
    strat = (df.loc[labeled, 'Bait'].astype(str) + '_' + y.loc[labeled].astype(str)).to_numpy()
    rows = []
    for cfg in GRID:
        model, _ = _fit(x, y, labeled, cfg, oob=True)
        rows.append(_internal_row(x, y, labeled, strat, model, cfg))
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def select(internal_csvs, out_csv):
    rows = []
    for ds, path in internal_csvs.items():
        g = pd.read_csv(path).set_index('config')
        v, sd = g['cv_logloss'], g['cv_logloss_sd']
        order = v.sort_values()
        best = order.index[0]
        rows.append({'dataset': ds, 'metric': 'cv_logloss', 'best_config': best,
                     'best': float(order.iloc[0]), 'best_sd': float(sd[best]),
                     'vanilla_config': DEPLOYED_LABEL, 'vanilla': float(v[DEPLOYED_LABEL]),
                     'vanilla_sd': float(sd[DEPLOYED_LABEL]),
                     'gain': float(v[DEPLOYED_LABEL] - order.iloc[0]),
                     'vanilla_rank': int(order.index.get_loc(DEPLOYED_LABEL)) + 1,
                     'n_configs': len(v), 'grid_min': float(v.min()), 'grid_max': float(v.max()),
                     'n_within_1sd_of_best': int((v <= order.iloc[0] + sd[best]).sum())})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def rf_params_for(label):
    cfg = CONFIG_BY_LABEL[label]
    return {k: cfg[k] for k in ('max_features', 'min_samples_leaf', 'class_weight', 'n_estimators')}


def compare(datasets, config_table, out_dir):
    rows = []
    for name, dv, dt in datasets:
        for gt in GROUND_TRUTHS:
            v = pd.read_csv(os.path.join(dv, f'auc_source_data_all_{gt}_all_methods.csv')).set_index('Method')
            t = pd.read_csv(os.path.join(dt, f'auc_source_data_all_{gt}_all_methods.csv')).set_index('Method')
            for m in v.index.intersection(t.index):
                row = {'dataset': name, 'ground_truth': gt, 'method': m}
                for depth in DEPTHS:
                    row[f'{depth} vanilla'] = float(v.loc[m, depth])
                    row[f'{depth} bestconfig'] = float(t.loc[m, depth])
                    row[f'{depth} diff'] = row[f'{depth} bestconfig'] - row[f'{depth} vanilla']
                    row[f'{depth} vanilla sd'] = float(v.loc[m, f'{depth} Std'])
                    row[f'{depth} bestconfig sd'] = float(t.loc[m, f'{depth} Std'])
                rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'bestconfig_vs_vanilla.csv'), index=False)
    _plot(table[table['method'] == 'WHISPER'], [n for n, _v, _t in datasets],
          os.path.join(out_dir, 'bestconfig_vs_vanilla.pdf'))


def _plot(whisper, datasets, out_pdf):
    fig, axes = plt.subplots(len(GROUND_TRUTHS), len(DEPTHS),
                             figsize=(1.35 * len(datasets) * len(DEPTHS) + 1.0,
                                      3.7 * len(GROUND_TRUTHS)), squeeze=False, sharex=True)
    for i, gt in enumerate(GROUND_TRUTHS):
        for j, depth in enumerate(DEPTHS):
            ax = axes[i][j]
            s = whisper[whisper['ground_truth'] == gt].set_index('dataset')
            s = s.reindex([d for d in datasets if d in s.index])
            x = np.arange(len(s), dtype=float)
            for run, dx in (('vanilla', -0.16), ('bestconfig', 0.16)):
                st = STYLE[run]
                ax.bar(x + dx, s[f'{depth} {run}'].to_numpy(float), width=0.32,
                       yerr=s[f'{depth} {run} sd'].to_numpy(float), color=st['color'],
                       capsize=3, error_kw=dict(elinewidth=1.2, ecolor='#333333'),
                       zorder=3, label=st['label'] if (i == 0 and j == 0) else None)
            ax.set_xticks(x)
            ax.set_xticklabels(list(s.index), rotation=35, ha='right', fontsize=13.5)
            ax.set_xlim(-0.6, len(s) - 0.4)
            for k in range(len(s) - 1):
                ax.axvline(k + 0.5, color='#d0d0cc', linewidth=0.8, zorder=0)
            ax.set_ylabel(depth, fontsize=16.5)
            ax.set_ylim(0, 1)
            ax.set_title(f'{GT_LABEL[gt]} — {depth}', fontsize=11.5, loc='left')
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', which='both', labelsize=14.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('WHISPER: defaults against the CV-selected configuration', fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
