"""Agreement between the variants of each method on BioID dataset 1
(Supplementary Fig. 1a)."""

import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

FDR_THRESHOLD = 0.01
TOP_N = 500


METHOD_COLUMNS = {
    'whisper': ('predicted_probability', 'FDR', 'Prey'),
    'SAINTq': ('AvgP', 'BFDR', 'PreyGene'),
    'SAINTexpress': ('AvgP', 'BFDR', 'PreyGene'),
    'limma': ('adj.P.Val', 'adj.P.Val', 'Protein'),
}


def _load(method, path):
    sep = '\t' if path.endswith(('.tsv', '.txt')) else ','
    df = pd.read_csv(path, sep=sep)
    score, fdr, prey = METHOD_COLUMNS[method]
    if method == 'whisper' and 'Prey' not in df.columns:
        df = df.rename(columns={'Protein': 'Prey'})
    if method in ('SAINTq', 'SAINTexpress') and 'PreyGene' not in df.columns:
        df = df.rename(columns={'Prey': 'PreyGene'})
    if fdr not in df.columns:
        df[fdr] = 1.0
    df['_score_'] = df[score].astype(float)
    df['_prey_'] = df[prey]
    df['_fdr_'] = df[fdr]
    return df


def _hits(df, bait):
    sub = df[(df['Bait'] == bait) & (df['_fdr_'] <= FDR_THRESHOLD)]
    return set(sub['_prey_'].dropna().astype(str))


def _top(df, bait):
    sub = df[df['Bait'] == bait]
    return set(sub.nlargest(TOP_N, '_score_')['_prey_'].dropna().astype(str)) if len(sub) else set()


def _ranks(df, bait):
    sub = df[df['Bait'] == bait][['_prey_', '_score_']].dropna()
    sub = sub.groupby('_prey_')['_score_'].max()
    sub.index = sub.index.astype(str)
    return sub


def _jaccard(a, b):
    if not a and not b:
        return np.nan
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def _spearman(s1, s2):
    common = s1.index.intersection(s2.index)
    if len(common) < 3:
        return np.nan
    return spearmanr(s1.loc[common], s2.loc[common])[0]


def _symmetric(m):
    for i in range(len(m)):
        m.iat[i, i] = 1.0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            a, b = m.iat[i, j], m.iat[j, i]
            if pd.isna(a) and not pd.isna(b):
                m.iat[i, j] = b
            elif pd.isna(b) and not pd.isna(a):
                m.iat[j, i] = a
    return m


def _average(mats):
    acc = None
    for m in mats:
        acc = m.copy().astype(float) if acc is None else acc.add(m, fill_value=np.nan)
    counts = None
    for m in mats:
        nn = (~m.isna()).astype(float)
        counts = nn if counts is None else counts.add(nn, fill_value=0)
    return acc / counts


def _heatmap(mat, title, out_pdf, diverging=False):
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat.astype(float), annot=True, fmt='.2f',
                cmap='coolwarm' if diverging else 'viridis',
                vmin=-1 if diverging else 0, vmax=1)
    plt.title(title, fontsize=13)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def _average_spearman_plot(mat, out_pdf, box=4.5, left=2.9, bottom=2.9, top=0.7, right=1.4):
    w, h = left + box + right, bottom + box + top
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, bottom / h, box / w, box / h])
    cax = fig.add_axes([(left + box + 0.25) / w, bottom / h, 0.2 / w, box / h])
    sns.heatmap(mat.astype(float), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                ax=ax, cbar_ax=cax, annot_kws={'size': 11})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('')
    cax.set_yticks([-1, -0.5, 0, 0.5, 1])
    cax.tick_params(labelsize=10)
    fig.text(0.5, 1 - 0.35 / h, 'Average across baits – Spearman (rank correlation)',
             ha='center', va='center', fontsize=13)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def run_method(method, variants, out_dir):

    out = os.path.join(out_dir, f'{method}_Variants_Similarity')
    os.makedirs(f'{out}/per_bait_csv', exist_ok=True)
    os.makedirs(f'{out}/per_bait_heatmaps', exist_ok=True)
    dfs = {name: _load(method, path) for name, path in variants.items()}
    modes = list(dfs)
    baits = list(pd.unique(next(iter(dfs.values()))['Bait']))
    j1_all, jn_all, sp_all = [], [], []
    for bait in baits:
        hits = {m: _hits(d, bait) for m, d in dfs.items()}
        tops = {m: _top(d, bait) for m, d in dfs.items()}
        ranks = {m: _ranks(d, bait) for m, d in dfs.items()}
        j1, jn, sp = (pd.DataFrame(index=modes, columns=modes, dtype=float) for _ in range(3))
        for m1, m2 in itertools.combinations(modes, 2):
            j1.loc[m1, m2] = j1.loc[m2, m1] = _jaccard(hits[m1], hits[m2])
            jn.loc[m1, m2] = jn.loc[m2, m1] = _jaccard(tops[m1], tops[m2])
            sp.loc[m1, m2] = sp.loc[m2, m1] = _spearman(ranks[m1], ranks[m2])
        j1, jn, sp = _symmetric(j1), _symmetric(jn), _symmetric(sp)
        j1.to_csv(f'{out}/per_bait_csv/{bait}_jaccard_1pct.csv')
        jn.to_csv(f'{out}/per_bait_csv/{bait}_jaccard_top{TOP_N}.csv')
        sp.to_csv(f'{out}/per_bait_csv/{bait}_spearman.csv')
        _heatmap(j1, f'{bait} – Jaccard (FDR ≤ {FDR_THRESHOLD:.2%})',
                 f'{out}/per_bait_heatmaps/{bait}_jaccard_1pct.pdf')
        _heatmap(jn, f'{bait} – Jaccard (Top {TOP_N})',
                 f'{out}/per_bait_heatmaps/{bait}_jaccard_top{TOP_N}.pdf')
        _heatmap(sp, f'{bait} – Spearman (rank correlation)',
                 f'{out}/per_bait_heatmaps/{bait}_spearman.pdf', diverging=True)
        j1_all.append(j1)
        jn_all.append(jn)
        sp_all.append(sp)
    avg_j1, avg_jn, avg_sp = _average(j1_all), _average(jn_all), _average(sp_all)
    avg_j1.to_csv(f'{out}/average_jaccard_1pct_across_baits.csv')
    avg_jn.to_csv(f'{out}/average_jaccard_top{TOP_N}_across_baits.csv')
    avg_sp.to_csv(f'{out}/average_spearman_across_baits.csv')
    _heatmap(avg_j1, f'Average across baits – Jaccard (FDR ≤ {FDR_THRESHOLD:.2%})',
             f'{out}/average_jaccard_1pct_across_baits.pdf')
    _heatmap(avg_jn, f'Average across baits – Jaccard (Top {TOP_N})',
             f'{out}/average_jaccard_top{TOP_N}_across_baits.pdf')
    _average_spearman_plot(avg_sp, f'{out}/average_spearman_across_baits.pdf')
