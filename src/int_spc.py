"""Intensity, spectral-count and combined models (Supplementary Fig. 1b)."""

import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_any_gene_in_gold(prey_str, gold_set):

    if pd.isna(prey_str):
        return 0
    return int(any(gene in gold_set for gene in str(prey_str).split(';')))

from sklearn.metrics import auc

warnings.filterwarnings('ignore')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

OUT_DIR = None

RUNS = {}         
RUN_ORDER = ['int only', 'spc only', 'int + spc']

GROUND_TRUTHS = {}  

DEPTHS = {'Total AUC': None, 'AUC@Top300': 300}
STEP_SIZE = 10
N_BOOTSTRAP = 100

RUN_COLORS = {'int only': '#2a78d6', 'spc only': '#eb6834',
              'int + spc': '#1baf7a'}
GRID_KW = dict(color='#d8d8d4', linewidth=0.6, zorder=0)
ZERO_KW = dict(linestyle=(0, (4, 3)), linewidth=1.2, color='#8a8a85')
UNIT_STYLE = {
    'pair': dict(color='#2a78d6', marker='o', label='pairs (optimistic)'),
    'bait': dict(color='#eb6834', marker='s', label='baits (conservative)'),
}


def tie_averaged_labels(score, true):

    score = np.asarray(score, dtype=float)
    score = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-score, kind='stable')
    s, t = score[order], np.asarray(true, dtype=float)[order]
    if len(s) == 0:
        return t
    block = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    return (np.bincount(block, weights=t) / np.bincount(block))[block]


def pr_auc_from_sorted(true_sorted, step_size=STEP_SIZE):
    n = len(true_sorted)
    if n == 0:
        return 0.0
    cum = np.cumsum(true_sorted)
    total_pos = cum[-1]
    if total_pos <= 0:
        return 0.0
    ks = np.arange(step_size, n + step_size, step_size)
    tp = cum[np.minimum(ks, n) - 1]
    return auc(tp / total_pos, tp / ks)


def load_aligned():
    frames = {}
    for name, path in RUNS.items():
        if not os.path.exists(path):
            raise SystemExit(f'{name}: missing {path}')
        frames[name] = pd.read_csv(path)
    keys = set.intersection(*[
        set(zip(d['Bait'].astype(str), d['Prey'].astype(str)))
        for d in frames.values()])
    order = [k for k in zip(frames['int only']['Bait'].astype(str),
                            frames['int only']['Prey'].astype(str))
             if k in keys]
    seen, uniq = set(), []
    for k in order:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    out = {}
    for name, d in frames.items():
        d = d.copy()
        d['_k'] = list(zip(d['Bait'].astype(str), d['Prey'].astype(str)))
        out[name] = d.drop_duplicates('_k').set_index('_k').reindex(uniq)
    return out, uniq


def bootstrap(aligned, truth, uniq, unit, n_bootstrap=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    scores = {n: d['predicted_probability'].to_numpy(dtype=float)
              for n, d in aligned.items()}
    n = len(uniq)
    if unit == 'bait':
        bait_of = np.array([k[0] for k in uniq])
        baits = sorted(set(bait_of))
        rows = {b: np.flatnonzero(bait_of == b) for b in baits}

    out = {name: {lab: np.empty(n_bootstrap) for lab in DEPTHS}
           for name in scores}
    for b in range(n_bootstrap):
        if unit == 'pair':
            idx = rng.integers(0, n, n)
        else:
            picked = rng.choice(len(baits), len(baits), replace=True)
            idx = np.concatenate([rows[baits[i]] for i in picked])
        for name, sc in scores.items():
            t = tie_averaged_labels(sc[idx], truth[idx])
            for lab, topn in DEPTHS.items():
                out[name][lab][b] = pr_auc_from_sorted(
                    t if topn is None else t[:topn])
    return out


def plot(point, diff, out_pdf):
    fig, axes = plt.subplots(2, len(DEPTHS),
                             figsize=(5.2 * len(DEPTHS), 7.4), squeeze=False)

    gts = list(GROUND_TRUTHS)
    x = np.arange(len(gts), dtype=float)
    for j, depth in enumerate(DEPTHS):
        ax = axes[0][j]
        for i, run in enumerate(RUN_ORDER):
            s = point[(point['model'] == run) & (point['depth'] == depth)]
            s = s.set_index('ground_truth').reindex(gts)
            pos = x - 0.26 + i * 0.26
            ax.bar(pos, s['AUC'], 0.24, color=RUN_COLORS[run], zorder=3,
                   label=run if j == 0 else None)
            ax.errorbar(pos, s['AUC'], yerr=s['AUC_sd'], fmt='none',
                        ecolor='#2a2a28', elinewidth=1.2, capsize=3, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(gts, fontsize=10)
        ax.set_ylabel(depth, fontsize=11)
        ax.set_title(depth, fontsize=11.5, loc='left')
        ax.set_ylim(0, max(0.2, float(point['AUC'].max()) * 1.2))

    refs = [r for r in RUN_ORDER if r != 'int only']
    ypos = {f'{r} — {g}': k for k, (r, g) in enumerate(
        [(r, g) for r in refs for g in gts][::-1])}
    for j, depth in enumerate(DEPTHS):
        ax = axes[1][j]
        ax.axvline(0, zorder=1, **ZERO_KW)
        for unit in ('pair', 'bait'):
            st = UNIT_STYLE[unit]
            for r in refs:
                for g in gts:
                    s = diff[(diff['model'] == r) & (diff['ground_truth'] == g)
                             & (diff['depth'] == depth)
                             & (diff['resampling_unit'] == unit)]
                    if s.empty:
                        continue
                    s = s.iloc[0]
                    y = ypos[f'{r} — {g}'] + (0.16 if unit == 'pair' else -0.16)
                    ax.errorbar([s['mean_difference']], [y],
                                xerr=[[s['mean_difference'] - s['ci_low']],
                                      [s['ci_high'] - s['mean_difference']]],
                                fmt=st['marker'], color=st['color'],
                                markersize=6.5, linewidth=0, elinewidth=1.5,
                                capsize=3.5, markeredgecolor='#fcfcfb',
                                markeredgewidth=1.2, zorder=3,
                                label=st['label'] if (j == 0 and r == refs[0]
                                                      and g == gts[0]) else None)
                    if s['ci_low'] <= 0 <= s['ci_high']:
                        ax.plot([s['mean_difference']], [y], marker=st['marker'],
                                color='#fcfcfb', markersize=3.4, zorder=4)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels(list(ypos), fontsize=9.5)
        ax.set_ylim(-0.6, len(ypos) - 0.4)
        ax.set_xlabel(f'Difference in {depth}\n(model − int only)', fontsize=10.5)
        for side in ('left',):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False)

    for row in axes:
        for ax in row:
            ax.grid(axis='y' if ax in list(axes[0]) else 'x', **GRID_KW)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=9.5)

    h0, l0 = axes[0][0].get_legend_handles_labels()
    h1, l1 = axes[1][0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, ncol=5, frameon=False, fontsize=9.5,
               loc='lower center', bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('Benchmarking BioID dataset 1, DDA: intensity, spectral '
                 'counts, and both', fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def _evaluate():
    os.makedirs(OUT_DIR, exist_ok=True)
    aligned, uniq = load_aligned()
    print(f'{len(uniq)} pairs shared by all three models')

    point_rows, diff_rows = [], []
    for gt_name, gt_file in GROUND_TRUTHS.items():
        g = pd.read_csv(gt_file)
  
        g = g[g['Prey'].isin({k[1] for k in uniq})]
        gold = {b: set(g[g['Bait'] == b]['Prey']) for b in {k[0] for k in uniq}}
        truth = np.array([is_any_gene_in_gold(k[1], gold.get(k[0], set()))
                          for k in uniq], dtype=float)
        print(f'  {gt_name}: {int(truth.sum())}/{len(truth)} pairs annotated '
              f'({100 * truth.mean():.1f}%)')

        boots = {u: bootstrap(aligned, truth, uniq, u) for u in ('pair', 'bait')}
        for run in RUN_ORDER:
            for depth in DEPTHS:
                a = boots['pair'][run][depth]
                point_rows.append({'model': run, 'ground_truth': gt_name,
                                   'depth': depth, 'AUC': float(a.mean()),
                                   'AUC_sd': float(a.std()),
                                   'n_pairs': len(uniq)})
        for unit, bt in boots.items():
            for run in RUN_ORDER:
                if run == 'int only':
                    continue
                for depth in DEPTHS:
                    d = bt[run][depth] - bt['int only'][depth]
                    diff_rows.append({
                        'model': run, 'reference': 'int only',
                        'ground_truth': gt_name, 'depth': depth,
                        'resampling_unit': unit,
                        'mean_difference': float(d.mean()),
                        'ci_low': float(np.percentile(d, 2.5)),
                        'ci_high': float(np.percentile(d, 97.5)),
                        'frac_positive': float((d > 0).mean()),
                        'n_bootstrap': N_BOOTSTRAP})

    point = pd.DataFrame(point_rows)
    diff = pd.DataFrame(diff_rows)
    point.to_csv(os.path.join(OUT_DIR, 'int_spc_auc_source_data.csv'),
                 index=False)
    diff.to_csv(os.path.join(OUT_DIR, 'int_spc_paired_differences.csv'),
                index=False)
    plot(point, diff, os.path.join(OUT_DIR, 'int_spc_auc.pdf'))

    print(f'\n{"model":<12}{"ground truth":<10}{"Total AUC":>18}'
          f'{"AUC@Top300":>18}')
    for run in RUN_ORDER:
        for gt in GROUND_TRUTHS:
            line = f'{run:<12}{gt:<10}'
            for depth in DEPTHS:
                s = point[(point['model'] == run) & (point['ground_truth'] == gt)
                          & (point['depth'] == depth)].iloc[0]
                line += f'{s["AUC"]:>12.4f} ±{s["AUC_sd"]:.3f}'
            print(line)

    print(f'\ndifference from the intensity model, 95% CI')
    for _i, r in diff.iterrows():
        mark = '' if r['ci_low'] <= 0 <= r['ci_high'] else '  *'
        print(f'  {r["model"]:<10}{r["ground_truth"]:<9}{r["depth"]:<12}'
              f'{r["resampling_unit"]:<6}{r["mean_difference"]:>+8.4f}  '
              f'[{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]{mark}')
    print('\n  * interval excludes zero')
    print(f'\nWrote {OUT_DIR}/int_spc_auc.pdf and two tables')


# ---------------------------------------------------------------- the int + spc model

SPC_SOURCE = ['log_fold_change', 'snr', 'mean_diff', 'replicate_fold_change_sd',
              'bait_control_sd_ratio', 'nonzero_reps']


def run(int_features_csv, spc_features_csv, int_scores, spc_scores, references, out_dir,
        initial_positives=15):
    global OUT_DIR
    from src.feature_engineering import FEATURE_COLUMNS
    from src import train_and_fdr as tf
    OUT_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)
    df_int = pd.read_csv(int_features_csv)
    spc = pd.read_csv(spc_features_csv)[['Bait', 'Prey'] + SPC_SOURCE]
    spc = spc.rename(columns={c: f'spc_{c}' for c in SPC_SOURCE})
    feats = df_int.merge(spc, on=['Bait', 'Prey'], how='inner')
    feats.to_csv(os.path.join(out_dir, 'aggregated_features_int_plus_spc.csv'), index=False)
    columns = FEATURE_COLUMNS + [f'spc_{c}' for c in SPC_SOURCE]
    saved = tf.FEATURE_COLUMNS
    tf.FEATURE_COLUMNS = columns            # the same pipeline on 14 features
    try:
        scored, _imp, _p = tf.run_training_and_fdr(feats, initial_positives)
    finally:
        tf.FEATURE_COLUMNS = saved
    combined = os.path.join(out_dir, 'FDR_BaitSpecific_Hierarchical_int_plus_spc.csv')
    scored.drop(columns=['global_cv_flag']).to_csv(combined, index=False)
    RUNS.clear()
    RUNS.update({'int only': int_scores, 'spc only': spc_scores, 'int + spc': combined})
    GROUND_TRUTHS.clear()
    GROUND_TRUTHS.update(references)
    _evaluate()
