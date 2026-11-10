"""Out-of-fold scoring (Supplementary Fig. 15)."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import benchmark as pr
from src.feature_engineering import FEATURE_COLUMNS
from src.train_and_fdr import monotone_q, strong_baits, weak_labels

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

N_SPLITS = 5
N_DECOY_REPLICATES = 30
DEPTHS = {'Total AUC': None, 'AUC@Top300': 300}
GT_LABEL = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}
DEPLOYED, OOF = 'WHISPER (full fit)', 'WHISPER (out of fold)'
STYLE = {
    DEPLOYED: dict(color='#2a78d6', marker='o', label='full fit'),
    OOF: dict(color='#eb6834', marker='s', label='scored out of fold'),
}


def _bagged_rf(oob=False):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    return BaggingClassifier(estimator=rf, n_estimators=100, random_state=42, oob_score=oob)


def train_cv(features_csv, initial_positives, out_csv, initial_negatives=200):
    df = pd.read_csv(features_csv).sort_values(['Bait', 'Prey']).reset_index(drop=True)
    eligible = df[~df['is_inert']]
    elig = ~df['is_inert'].to_numpy()
    np.random.seed(42)
    x_real = df[FEATURE_COLUMNS].values
    y = weak_labels(df, eligible, strong_baits(df, eligible), initial_positives, initial_negatives)
    labeled = y[y != 0].index
    df['weak_label'] = y

    reps = []
    for rep in range(N_DECOY_REPLICATES):
        parts = []
        for i, bait in enumerate(eligible['Bait'].unique()):
            np.random.seed(42 + rep * 1000 + i)
            d = eligible[eligible['Bait'] == bait].copy()
            for col in FEATURE_COLUMNS:
                d[col] = np.random.permutation(d[col].values)
            parts.append(d)
        allp = pd.concat(parts).sort_index()
        assert allp.index.equals(eligible.index), 'decoy rows lost alignment with real rows'
        reps.append(allp[FEATURE_COLUMNS].values)
    x_decoy = np.vstack(reps)

    strat = (df.loc[labeled, 'Bait'].astype(str) + '_' + y.loc[labeled].astype(str)).values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_of_row = pd.Series(-1, index=df.index)
    real_by_fold = np.empty((N_SPLITS, len(df)))
    decoy_by_fold = np.empty((N_SPLITS, N_DECOY_REPLICATES * len(eligible)))
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(labeled)), strat)):
        train_rows, test_rows = labeled[tr], labeled[te]
        fold_of_row.loc[test_rows] = fold
        sc = StandardScaler()
        model = _bagged_rf()
        model.fit(sc.fit_transform(x_real[train_rows]), y.loc[train_rows])
        real_by_fold[fold] = model.predict_proba(sc.transform(x_real))[:, 1]
        decoy_by_fold[fold] = model.predict_proba(sc.transform(x_decoy))[:, 1]

    held_out = fold_of_row.values
    is_lab = held_out >= 0
    pos = np.arange(len(df))
    cv_real = real_by_fold.mean(axis=0)
    cv_real[is_lab] = real_by_fold[held_out[is_lab], pos[is_lab]]
    held_rep = np.tile(held_out[elig], N_DECOY_REPLICATES)
    dpos = np.arange(N_DECOY_REPLICATES * len(eligible))
    lab_rep = held_rep >= 0
    cv_decoy = decoy_by_fold.mean(axis=0)
    cv_decoy[lab_rep] = decoy_by_fold[held_rep[lab_rep], dpos[lab_rep]]
    df['predicted_probability'] = cv_real
    df['cv_fold'] = held_out
    for k in range(N_SPLITS):
        df[f'prob_fold{k}'] = real_by_fold[k]

    sc = StandardScaler()
    full = _bagged_rf(oob=True)
    full.fit(sc.fit_transform(x_real[labeled]), y.loc[labeled])
    df['predicted_probability_simple'] = full.predict_proba(sc.transform(x_real))[:, 1]
    simple_decoy = full.predict_proba(sc.transform(x_decoy))[:, 1]

    df['FDR'] = monotone_q(cv_real[elig], cv_decoy, N_DECOY_REPLICATES, cv_real)
    simple = df['predicted_probability_simple'].values
    df['FDR_simple'] = monotone_q(simple[elig], simple_decoy, N_DECOY_REPLICATES, simple)
    thr = np.percentile(df['global_cv'], 25)
    df['global_cv_flag'] = np.where(df['global_cv'] <= thr, 'likely background', '')
    df.to_csv(out_csv, index=False)


def evaluate(datasets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cfg, cv_path in datasets:
        rows.extend(evaluate_dataset(cfg, cv_path))
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(out_dir, 'cv_per_fold_auc.csv'), index=False)
    summary = summarise(per_fold)
    summary.to_csv(os.path.join(out_dir, 'cv_per_fold_summary.csv'), index=False)
    plot(summary, os.path.join(out_dir, 'cv_per_fold.pdf'))


def auc_of(score, true):
    score = np.where(np.isnan(score), -np.inf, score)
    t = pr.tie_averaged_labels(score, true)
    return {lab: pr._pr_auc_from_sorted(t if n is None else t[:n])
            for lab, n in DEPTHS.items()}


def evaluate_dataset(cfg, cv_path):
    name = cfg['name']
    cv = pd.read_csv(cv_path)
    fold_cols = [f'prob_fold{k}' for k in range(N_SPLITS)]
    missing = [c for c in fold_cols if c not in cv.columns]
    if missing:
        raise SystemExit(f'{name}: {cv_path} has no {missing}. '
                         f'Run the cross_validation step first.')

    features_df, frames = pr.load_new_methods(cfg, None)
    frames.update(pr.load_old_methods(cfg, features_df))
    frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')

    if len(cv) != len(features_df) or not (
            cv['Bait'].astype(str).values
            == features_df['Bait'].astype(str).values).all():
        raise SystemExit(f'{name}: CV and deployed tables are not row-aligned')
    gap = np.abs(cv['predicted_probability_simple'].values
                 - features_df['predicted_probability'].values).max()
    if gap > 1e-9:
        raise SystemExit(f'{name}: the CV run full fit differs from the '
                         f'deployed model (max {gap:.3g})')
    for c in fold_cols + ['cv_fold']:
        features_df[c] = cv[c].values

    present = [m for m in pr.ALL_METHODS if m in frames]
    baits = list(features_df['Bait'].unique())
    identified = set(features_df['Prey'].astype(str))
    print(f'\n{name}: {len(features_df)} pairs, methods {", ".join(present)}')

    rows = []
    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(identified)]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}

        keyed = {}
        for m in present:
            df, prey_col, score_col = frames[m]
            d = df.copy()
            d['_k'] = list(zip(d['Bait'].astype(str), d[prey_col].astype(str)))
            keyed[m] = (d.drop_duplicates('_k').set_index('_k'), score_col)
        wd = keyed['WHISPER'][0]
        order = [k for k in wd.index
                 if all(k in keyed[m][0].index for m in present)]
        w = wd.reindex(order)
        true = np.array([pr.is_any_gene_in_gold(k[1], gold.get(k[0], set()))
                         for k in order], dtype=float)
        fold = w['cv_fold'].to_numpy(dtype=int)

        scores = {m: pd.to_numeric(keyed[m][0].reindex(order)[sc],
                                   errors='coerce').to_numpy(dtype=float)
                  for m, (_d, sc) in keyed.items()}

        for k in range(N_SPLITS):
            test = (fold == k) | (fold == -1)
            entries = {DEPLOYED: scores['WHISPER'],
                       OOF: w[f'prob_fold{k}'].to_numpy(dtype=float)}
            entries.update({m: s for m, s in scores.items() if m != 'WHISPER'})
            for m, s in entries.items():
                a = auc_of(s[test], true[test])
                rows.append({'dataset': name, 'ground_truth': gt_source,
                             'fold': k, 'method': m, 'n_test': int(test.sum()),
                             'n_positive': int(true[test].sum()), **a})
        sizes = [int(((fold == k) | (fold == -1)).sum()) for k in range(N_SPLITS)]
        print(f'  {GT_LABEL[gt_source]}: {len(order)} pairs, fold test sets of '
              f'{min(sizes)}-{max(sizes)} pairs')
    return rows


def summarise(per_fold):
    g = per_fold.groupby(['dataset', 'ground_truth', 'method'])
    out = g[list(DEPTHS)].agg(['mean', 'std'])
    out.columns = [f'{a} {b}' for a, b in out.columns]
    return out.reset_index()


def plot(summary, out_pdf):
    datasets = list(dict.fromkeys(summary['dataset']))
    short = datasets
    gts = list(GT_LABEL)
    fig, axes = plt.subplots(len(gts), len(DEPTHS),
                             figsize=(1.35 * len(datasets) * len(DEPTHS) + 1.0,
                                      3.7 * len(gts)),
                             squeeze=False, sharex=True)
    for i, gt in enumerate(gts):
        for j, depth in enumerate(DEPTHS):
            ax = axes[i][j]
            x = np.arange(len(datasets), dtype=float)
            for run, dx in ((DEPLOYED, -0.16), (OOF, 0.16)):
                st = STYLE[run]
                s = summary[(summary['ground_truth'] == gt)
                            & (summary['method'] == run)].set_index('dataset')
                s = s.reindex(datasets)
                ax.errorbar(x + dx, s[f'{depth} mean'], yerr=s[f'{depth} std'],
                            fmt=st['marker'], color=st['color'], markersize=7,
                            linewidth=0, elinewidth=1.4, capsize=3,
                            markeredgecolor='#fcfcfb', markeredgewidth=1.2,
                            zorder=4,
                            label=st['label'] if (i == 0 and j == 0) else None)
            ax.set_xticks(x)
            ax.set_xticklabels(short, rotation=35, ha='right', fontsize=13.5)
            ax.set_xlim(-0.6, len(datasets) - 0.4)
            ax.set_ylabel(depth, fontsize=16.5)
            ax.set_ylim(0, 1)
            ax.set_title(f'{GT_LABEL[gt]} — {depth}', fontsize=11.5, loc='left')
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', which='both', labelsize=14.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=10,
               loc='lower center', bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('WHISPER: full fit against out-of-fold scoring, evaluated on '
                 'each fold\'s test set', fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
