"""Bounds on the false discovery proportion (Supplementary Figs. 12 and 13)."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from sklearn.preprocessing import StandardScaler

from src.benchmark import load_uniprot_to_gene, make_to_gene
from src.feature_engineering import FEATURE_COLUMNS
from src.train_and_fdr import build_model, fit, monotone_q

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

BAIT_SIDE = {'PDHA1': 'mito_matrix', 'EGFR': 'luminal'}
MITO_LUMEN_TERMS = ['GO:0005759', 'GO:0005743']                 # matrix, inner membrane
SECRETORY_LUMEN_TERMS = ['GO:0005788', 'GO:0005796', 'GO:0043202']  # ER, Golgi, lysosome lumen
PEROXISOME_TERMS = ['GO:0005782']                               # peroxisomal matrix
FACING_TERMS = ['GO:0005829', 'GO:0005634', 'GO:0005886', 'GO:0005856', 'GO:0005741',
                'GO:0005758', 'GO:0005789', 'GO:0000139', 'GO:0010008', 'GO:0005765',
                'GO:0005778', 'GO:0005635']
MITOCHONDRION = 'GO:0005739'
HPA_LUMEN_OK = {'Mitochondria', 'Endoplasmic reticulum', 'Golgi apparatus', 'Vesicles',
                'Lysosomes', 'Peroxisomes', 'Endosomes'}
BIOTINYLATED = {'PC', 'PCCA', 'PCCB', 'MCCC1', 'MCCC2', 'ACACA', 'ACACB'}
METHODS = ['WHISPER', 'limma', 'SAINTq', 'SAINTexpress']
COLORS = {'WHISPER': '#003f5c', 'limma': '#d62728', 'SAINTq': '#ffb347',
          'SAINTexpress': '#ff7f0e'}
REF_NAMES = ('GO:CC', 'BioGRID')
REF_KEYS = {'GO:CC': 'go_large', 'BioGRID': 'bg_large'}
REF_STYLES = {'GO:CC': (0, (5, 2)), 'BioGRID': (0, (1, 1.5))}
Q_GRID = [round(q, 3) for q in np.arange(0.005, 0.2001, 0.005)]

NULLS = ['individual', 'grouped', 'permutation']
NULL_TITLES = {'individual': 'WHISPER (individual)', 'grouped': 'WHISPER (grouped)',
               'permutation': 'WHISPER (label permutation)'}
NULL_COLORS = {'individual': '#003f5c', 'grouped': '#8b1a1a', 'permutation': '#7b2d8e'}
SCHEMES = {
    'individual': {f: [f] for f in FEATURE_COLUMNS},
    'grouped': {'enrichment': ['log_fold_change', 'snr'],
                'sd_measures': ['replicate_fold_change_sd', 'bait_control_sd_ratio'],
                'central_tendency': ['mean_diff', 'median_diff'],
                'bait_cv': ['bait_cv'], 'zero_or_neg_fc': ['zero_or_neg_fc']},
}
N_REPLICATES = 30
N_LABEL_PERMUTATIONS = 30
LABEL_PERM_SEED_BASE = 9000
N_NEGATIVES = 200
X_MAX = 0.1
Y_ZERO = 4e-4
X_LOG_MIN = 0.005
X_LOG_TICKS = [0.005, 0.01, 0.02, 0.05, 0.1]


# ---------------------------------------------------------------- lower-bound list

def unreachable_genes(gaf_file, obo_file, hpa_file):
    """Listed genes for each bait side."""
    from goatools.obo_parser import GODag
    dag = GODag(obo_file, optional_attrs={'relationship'}, prt=None)

    def expand(terms):
        out, stack = set(terms), list(terms)
        while stack:
            t = dag[stack.pop()]
            for c in list(t.children) + list(t.relationship_rev.get('part_of', [])):
                if c.id not in out:
                    out.add(c.id)
                    stack.append(c.id)
        return out

    gaf = pd.read_csv(gaf_file, sep='\t', comment='!', header=None, usecols=[2, 3, 4, 6, 8],
                      names=['gene', 'qualifier', 'go', 'evidence', 'aspect'], dtype=str)
    gaf = gaf[(gaf['aspect'] == 'C') & ~gaf['qualifier'].str.startswith('NOT')]
    curated = gaf[gaf['evidence'] != 'IEA']

    def annotated(terms, frame):
        return set(frame.loc[frame['go'].isin(expand(terms)), 'gene'])

    in_mito_lumen = annotated(MITO_LUMEN_TERMS, curated)
    in_other_lumen = annotated(SECRETORY_LUMEN_TERMS, curated) | annotated(PEROXISOME_TERMS, curated)
    in_peroxisome = annotated(PEROXISOME_TERMS, curated)
    faces = annotated(FACING_TERMS, gaf)
    mito = annotated([MITOCHONDRION], gaf)
    hpa = pd.read_csv(hpa_file, sep='\t')
    locs = (hpa['Main location'].fillna('') + ';' + hpa['Additional location'].fillna('')).str.split(';')
    for g, loc in zip(hpa['Gene name'].astype(str), locs):
        loc = {x for x in loc if x}
        if loc - HPA_LUMEN_OK:
            faces.add(g)
        if 'Mitochondria' in loc:
            mito.add(g)
    return {'cytosolic': (in_mito_lumen | in_other_lumen) - faces - BIOTINYLATED,
            'mito_matrix': in_other_lumen - faces - mito - BIOTINYLATED,
            'luminal': (in_mito_lumen | in_peroxisome) - faces - BIOTINYLATED}


def _all_in(prey, s):
    return all(g in s for g in str(prey).split(';'))


def _any_in(prey, s):
    return any(g in s for g in str(prey).split(';'))


def _references(cfg, identified):
    refs = {n: pd.read_csv(cfg['references'][REF_KEYS[n]]) for n in REF_NAMES}
    gold = {n: {b: set(x['Prey'].astype(str)) for b, x in
                g[g['Prey'].astype(str).isin(identified)].groupby('Bait')}
            for n, g in refs.items()}
    partners = {}
    for g in refs.values():
        for b, x in g.groupby('Bait'):
            partners.setdefault(b, set()).update(x['Prey'].astype(str))
    return gold, partners


def _bound_rows(df, listed, gold, label_key, label):
    df = df.copy()
    df['listed'] = [_all_in(p, listed.get(b, set())) for b, p in zip(df['Bait'], df['Prey'])]
    for ref in REF_NAMES:
        df[ref] = [_any_in(p, gold[ref].get(b, set())) for b, p in zip(df['Bait'], df['Prey'])]
    rows = []
    for q in Q_GRID:
        called = df[df['FDR'] <= q + 1e-12]
        n = len(called)
        row = {label_key: label, 'nominal_q': q, 'calls': n,
               'listed_calls': int(called['listed'].sum()),
               'lower': called['listed'].mean() if n else np.nan}
        for ref in REF_NAMES:
            row[f'upper_{ref}'] = 1 - called[ref].mean() if n else np.nan
        rows.append(row)
    return rows


def _to_gene(cfg):
    return make_to_gene(load_uniprot_to_gene(cfg['gaf_file'])) if cfg.get('map_identifiers') else None


# ---------------------------------------------------------------- Sup. Fig. 12

def run_methods(cfg, lists):
    """cfg: scores, methods, references, results_dir, map_identifiers, gaf_file."""
    out_dir = os.path.join(cfg['results_dir'], 'fdr_bounds_contaminants_lumen')
    os.makedirs(out_dir, exist_ok=True)
    m = cfg.get('methods', {})
    calls = {'WHISPER': pd.read_csv(cfg['scores'])[['Bait', 'Prey', 'FDR']]}
    if m.get('limma'):
        lm = pd.read_csv(m['limma'], sep='\t').rename(columns={'Protein': 'Prey'})
        calls['limma'] = pd.DataFrame({'Bait': lm['Bait'], 'Prey': lm['Prey'],
                                       'FDR': lm['adj.P.Val.one_sided']})
    for name, prey in (('SAINTq', 'Prey'), ('SAINTexpress', 'PreyGene')):
        if m.get(name.lower()):
            d = pd.read_csv(m[name.lower()], sep='\t')
            calls[name] = pd.DataFrame({'Bait': d['Bait'], 'Prey': d[prey], 'FDR': d['BFDR']})
    to_gene = _to_gene(cfg)
    if to_gene:
        for d in calls.values():
            d['Prey'] = d['Prey'].astype(str).map(to_gene)

    whisper = calls['WHISPER']
    identified = set(whisper['Prey'].astype(str))
    baits = sorted(whisper['Bait'].unique())
    gold, partners = _references(cfg, identified)
    listed = {b: lists[BAIT_SIDE.get(b, 'cytosolic')] - {b} - partners.get(b, set())
              for b in baits}
    detected = {g for p in identified for g in p.split(';')}
    pd.DataFrame([dict(Bait=b, side=BAIT_SIDE.get(b, 'cytosolic'), gene=g, detected=g in detected)
                  for b in baits for g in sorted(listed[b])]).to_csv(
        os.path.join(out_dir, 'contaminant_list.csv'), index=False)

    rows = []
    for method in [x for x in METHODS if x in calls]:
        rows += _bound_rows(calls[method], listed, gold, 'method', method)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'fdr_bounds_contaminants.csv'), index=False)
    for log in (False, True):
        plot_bounds(res, 'method', [x for x in METHODS if x in set(res['method'])], COLORS,
                    {x: x for x in METHODS}, cfg['dataset'],
                    os.path.join(out_dir, f'fdr_bounds_contaminants{"_logy" if log else ""}.pdf'),
                    log=log, hits=False)


# ---------------------------------------------------------------- Sup. Fig. 13

def _shuffle_scores(df, model, scaler, categories, seed_base=42):
    df = df[~df['is_inert']]
    out = []
    for rep in range(N_REPLICATES):
        for i, bait in enumerate(df['Bait'].unique()):
            np.random.seed(seed_base + rep * 1000 + i)
            decoy = df[df['Bait'] == bait].copy()
            n = len(decoy)
            for cols in categories.values():
                order = np.random.permutation(n)
                for col in cols:
                    decoy[col] = decoy[col].values[order]
            out.append(model.predict_proba(scaler.transform(decoy[FEATURE_COLUMNS].values))[:, 1])
    return np.concatenate(out)


def _label_permutation_scores(df, train_baits, n_positives):
    df = df[~df['is_inert']].reset_index(drop=True)
    x = df[FEATURE_COLUMNS].values
    out = []
    for k in range(N_LABEL_PERMUTATIONS):
        rng = np.random.default_rng(LABEL_PERM_SEED_BASE + k)
        y = pd.Series(0, index=df.index)
        for bait in train_baits:
            rows = df[df['Bait'] == bait]
            if len(rows) < n_positives + N_NEGATIVES:
                continue
            picked = rng.choice(rows.index.to_numpy(), n_positives + N_NEGATIVES, replace=False)
            y.loc[picked[:n_positives]] = 1
            y.loc[picked[n_positives:]] = -1
        labeled = y[y != 0].index
        if labeled.empty:
            continue
        sc = StandardScaler()
        model = build_model(seed=LABEL_PERM_SEED_BASE + k, n_jobs=-1)
        model.fit(sc.fit_transform(x[labeled]), y.loc[labeled])
        out.append(model.predict_proba(sc.transform(x))[:, 1])
    return np.concatenate(out)


def null_q(features_csv, initial_positives):
    """WHISPER refitted, and the q of every pair under each null."""
    scored, model, scaler, y = fit(pd.read_csv(features_csv), initial_positives, N_NEGATIVES,
                                   n_jobs=-1)
    elig = ~scored['is_inert'].to_numpy()
    probs = scored['predicted_probability'].to_numpy()
    strong = sorted(scored.loc[y != 0, 'Bait'].unique())
    out = scored[['Bait', 'Prey', 'predicted_probability']].copy()
    for null in NULLS:
        decoys = (_label_permutation_scores(scored, strong, initial_positives)
                  if null == 'permutation' else _shuffle_scores(scored, model, scaler, SCHEMES[null]))
        out[f'FDR_{null}'] = monotone_q(probs[elig], decoys, N_REPLICATES, probs)
    return out


def run_nulls(cfg, lists):
    """cfg: features, initial_positives, references, results_dir, map_identifiers,
    gaf_file."""
    out_dir = os.path.join(cfg['results_dir'], 'fdr_bounds_nulls')
    os.makedirs(out_dir, exist_ok=True)
    q = null_q(cfg['features'], cfg['initial_positives'])
    q.to_csv(os.path.join(out_dir, 'whisper_null_q.csv'), index=False)
    calls = q.copy()
    to_gene = _to_gene(cfg)
    if to_gene:
        calls['Prey'] = calls['Prey'].astype(str).map(to_gene)
    identified = set(calls['Prey'].astype(str))
    baits = sorted(calls['Bait'].unique())
    gold, partners = _references(cfg, identified)
    listed = {b: lists[BAIT_SIDE.get(b, 'cytosolic')] - {b} - partners.get(b, set())
              for b in baits}
    rows = []
    for null in NULLS:
        rows += _bound_rows(calls.rename(columns={f'FDR_{null}': 'FDR'}), listed, gold,
                            'null', null)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'fdr_bounds_nulls.csv'), index=False)
    for log in (False, True):
        plot_bounds(res, 'null', NULLS, NULL_COLORS, NULL_TITLES, cfg['dataset'],
                    os.path.join(out_dir, f'fdr_bounds_nulls{"_logy" if log else ""}.pdf'),
                    log=log, hits=True)


# ---------------------------------------------------------------- figures

def _zero_floor(v, log):
    return v.where(~(v <= 0), Y_ZERO) if log else v


def _log_axes(ax, top):
    ax.set_xscale('log')
    ax.set_xlim(X_LOG_MIN, top)
    ax.set_xticks(X_LOG_TICKS)
    ax.set_xticklabels([f'{t:g}' for t in X_LOG_TICKS])
    ax.set_yscale('log')
    ax.set_ylim(Y_ZERO / 1.6, 1)
    ticks = [Y_ZERO, 1e-3, 1e-2, 1e-1, 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels(['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
    ax.minorticks_off()
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    mid = np.sqrt(Y_ZERO * 1e-3)
    ax.plot([0, 0], [mid / 1.12, mid * 1.12], transform=tr, color='white', linewidth=3,
            clip_on=False, zorder=5)
    for y in (mid / 1.12, mid * 1.12):
        ax.plot([-0.025, 0.025], [y / 1.08, y * 1.08], transform=tr, color='black',
                linewidth=0.8, clip_on=False, zorder=6)


def plot_bounds(res, key, groups, colors, titles, dataset, out_pdf, log=False, hits=False):
    """One panel per method (Sup. 12) or, with hits, a bounds panel and a
    number-of-hits panel per null (Sup. 13); x axis nominal FDR up to 0.1."""
    res = res[res['nominal_q'] <= X_MAX + 1e-12]
    ncol = len(groups) * (2 if hits else 1)
    fig, axes = plt.subplots(1, ncol, figsize=((5.4 if hits else 2.7) * len(groups), 3.6),
                             sharey=not hits, squeeze=False)
    for i, g in enumerate(groups):
        ax = axes[0][2 * i if hits else i]
        r = res[res[key] == g]
        col = colors[g]
        diag = np.linspace(X_LOG_MIN if log else 0, X_MAX, 200)
        ax.plot(diag, diag, color='#8a8a85', linestyle=(0, (4, 3)), linewidth=1)
        upper = _zero_floor(r[[f'upper_{ref}' for ref in REF_NAMES]].min(axis=1), log)
        ax.fill_between(r['nominal_q'], _zero_floor(r['lower'], log), upper, color=col,
                        alpha=0.10, linewidth=0)
        ax.plot(r['nominal_q'], _zero_floor(r['lower'], log), color=col, linewidth=2)
        for ref in REF_NAMES:
            ax.plot(r['nominal_q'], _zero_floor(r[f'upper_{ref}'], log), color=col,
                    linewidth=2, linestyle=REF_STYLES[ref])
        ax.set_xlim(0, X_MAX)
        ax.set_xticks([0, 0.05, 0.1])
        if log:
            _log_axes(ax, X_MAX)
        else:
            ax.set_ylim(0, 1)
        ax.set_ylabel('Estimated FDP', fontsize=12)
        panels = [ax]
        if hits:
            hx = axes[0][2 * i + 1]
            hx.plot(r['nominal_q'], r['calls'], color=col, linewidth=2)
            hx.set_xlim(0, X_MAX)
            if log:
                hx.set_xscale('log')
                hx.set_xlim(X_LOG_MIN, X_MAX)
                hx.set_xticks(X_LOG_TICKS)
                hx.set_xticklabels([f'{t:g}' for t in X_LOG_TICKS])
                hx.minorticks_off()
            hx.set_ylim(0, res['calls'].max() * 1.05)
            hx.set_ylabel('Number of hits', fontsize=12)
            panels.append(hx)
        for a in panels:
            a.set_title(titles[g], fontsize=11, fontweight='normal', fontstyle='normal')
            a.set_xlabel('Nominal FDR', fontsize=12)
            a.tick_params(labelsize=10.5)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
    handles = [Line2D([], [], color='#8a8a85', linestyle=(0, (4, 3)), linewidth=1, label='Nominal'),
               Line2D([], [], color='#555555', linewidth=2, label='Lower bound (lumen proteins)')]
    handles += [Line2D([], [], color='#555555', linewidth=2, linestyle=REF_STYLES[ref],
                       label=f'Upper bound ({ref})') for ref in REF_NAMES]
    fig.legend(handles=handles, ncol=len(handles) if hits else 2, frameon=False, fontsize=9,
               loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(dataset, fontsize=13, fontweight='normal', fontstyle='normal')
    fig.tight_layout(rect=[0, 0.07 if hits else 0.13, 1, 1])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
