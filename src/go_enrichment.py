"""GO:CC enrichment and UpSet of high-confidence preys (Fig. 6a-c,
Supplementary Tables 11-15)."""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

MAX_TERM_SIZE = 1000
EMPTY_COLUMNS = ['source', 'native', 'name', 'description', 'p_value', 'significant',
                 'term_size', 'query_size', 'intersection_size', 'intersection',
                 'effective_domain_size', 'precision', 'recall', 'Bait', 'Method']


def get_sig_preys(df, bait, method, fdr_threshold):
    if method == 'limma':
        d = df[(df['Bait'] == bait) & (df['adj.P.Val.one_sided'] <= fdr_threshold)]
        return set(d['Prey'].dropna().astype(str))
    if method in ('SAINTq', 'SAINTexpress'):
        d = df[(df['Bait'] == bait) & (df['BFDR'] <= fdr_threshold)]
        return set(d['PreyGene'].dropna().astype(str))
    d = df[(df['Bait'] == bait) & (df['FDR'] <= fdr_threshold)]
    return set(d['Prey'].dropna().astype(str))


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return cm.colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', cmap(np.linspace(minval, maxval, n)))


def plot_dotplot(enrich_df, bait, out_pdf, top_terms=10, max_term_length=40,
                 max_term_size=MAX_TERM_SIZE):
    df = enrich_df.copy()
    if 'term_size' in df.columns:
        df = df[df['term_size'] <= max_term_size]
    df['name'] = df['name'].apply(lambda x: x[:max_term_length] + '...'
                                  if isinstance(x, str) and len(x) > max_term_length else x)
    top = df.groupby('Method', group_keys=False).apply(
        lambda x: x.nsmallest(top_terms, 'p_value')).reset_index(drop=True)
    if top.empty:
        return
    top['neglog10_p'] = -np.log10(top['p_value'].replace(0, np.nextafter(0, 1)))
    s_min, s_max = 50, 300
    vmin = max(float(top['intersection_size'].min()), 0.0)
    vmax = float(top['intersection_size'].max()) or 1.0
    cmap = _truncate_colormap(cm.get_cmap('Oranges'), 0.35, 1.0)
    order = sorted(top['Method'].unique())
    top['Method_pos'] = top['Method'].map({m: i for i, m in enumerate(order)})

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=top, x='name', y='Method_pos', size='intersection_size',
                    hue='neglog10_p', sizes=(s_min, s_max), palette=cmap, ax=ax,
                    edgecolor='none')
    ax.set_xlabel('GO Term')
    ax.set_ylabel('Method')
    ax.set_title(f'Enriched GO Terms for {bait}', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.5, len(order) - 0.5)
    if ax.legend_:
        ax.legend_.remove()

    def area(v):
        return s_max if vmax == vmin else s_min + (float(v) - vmin) / (vmax - vmin) * (s_max - s_min)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                          markersize=np.sqrt(area(v)), label=f'{int(v)}') for v in (vmin, vmax)]
    ax.legend(handles=handles, title='Intersection Size', loc='upper left',
              bbox_to_anchor=(1.1, 1.0), frameon=True)
    plt.subplots_adjust(right=0.86)
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=float(top['neglog10_p'].min()),
                                              vmax=float(top['neglog10_p'].max())), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01).set_label(
        r'$-\log_{10}$ (Adjusted P-Value)', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close()


def run(cfg):

    from gprofiler import GProfiler
    from upsetplot import UpSet, from_memberships

    out = cfg['outdir']
    for sub in ('dotplots', 'upset', 'enrichment_tables/overall_enrichment',
                'enrichment_tables/upset_enrichment'):
        os.makedirs(f'{out}/{sub}', exist_ok=True)
    thr = cfg.get('fdr_threshold', 0.01)

    whisper = pd.read_csv(cfg['fdr_output'])
    limma = pd.read_csv(cfg['limma'], sep='\t').rename(columns={'Protein': 'Prey'})
    saintq = pd.read_csv(cfg['saintq'], sep='\t').rename(columns={'Prey': 'PreyGene'})
    saintex = pd.read_csv(cfg['saintexpress'], sep='\t')
    all_dfs = {'puppi': whisper, 'limma': limma, 'SAINTq': saintq, 'SAINTexpress': saintex}

    gp = GProfiler(return_dataframe=True)
    background = sorted(whisper['Prey'].dropna().astype(str).unique())

    def profile(preys, bait, method):
        if len(preys) < 3:
            return None
        try:
            res = gp.profile(organism='hsapiens', query=list(preys), sources=['GO:CC'],
                             user_threshold=0.05, domain_scope='custom', background=background)
            res['Bait'] = bait
            res['Method'] = method
            return res
        except Exception as exc:
            print(f'    g:Profiler failed for {bait}/{method}: {exc}')
            return None

    for bait in whisper['Bait'].unique():
        preys = {m: get_sig_preys(df, bait, m, thr) for m, df in all_dfs.items()}

        overall = {m: r for m, p in preys.items()
                   if (r := profile(p, bait, m)) is not None and not r.empty}
        if overall:
            with pd.ExcelWriter(f'{out}/enrichment_tables/overall_enrichment/{bait}_gprofiler.xlsx') as w:
                for m, df in overall.items():
                    df.to_excel(w, sheet_name=m, index=False)
            plot_dotplot(pd.concat(overall.values(), ignore_index=True), f'{bait}_overall',
                         f'{out}/dotplots/{bait}_overall_gprofiler_dotplot.pdf')

        union = set.union(*preys.values())
        memberships = [tuple(m for m, p in preys.items() if prey in p) for prey in union]
        memberships = [t for t in memberships if t]
        if not memberships:
            continue
        plt.figure(figsize=(4, 10))
        UpSet(from_memberships(memberships), subset_size='count', show_counts=True,
              sort_by='degree').plot()
        plt.suptitle(f'{bait}: Significant Prey Overlaps (FDR ≤ {thr})', fontsize=14)
        plt.savefig(f'{out}/upset/{bait}_upset.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        upset_enrich = {}
        base = f'{out}/enrichment_tables/upset_enrichment/{bait}'
        with pd.ExcelWriter(f'{base}_upset_gprofiler.xlsx') as w, \
                pd.ExcelWriter(f'{base}_upset_preys.xlsx') as pw:
            for combo in sorted(set(memberships)):
                name = '+'.join(combo)
                sheet = name if len(name) <= 31 else name[:28] + '...'
                exclusive = {p for p in union
                             if {m for m, s in preys.items() if p in s} == set(combo)}
                pd.DataFrame(sorted(exclusive), columns=['Prey']).to_excel(pw, sheet_name=sheet,
                                                                          index=False)
                res = profile(exclusive, bait, name)
                if res is not None and not res.empty:
                    res.to_excel(w, sheet_name=sheet, index=False)
                    upset_enrich[name] = res
                else:
                    empty = pd.DataFrame(columns=EMPTY_COLUMNS)
                    empty['Bait'] = bait
                    empty['Method'] = name
                    empty.to_excel(w, sheet_name=sheet, index=False)
        if upset_enrich:
            plot_dotplot(pd.concat(upset_enrich.values(), ignore_index=True), f'{bait}_upset',
                         f'{out}/dotplots/{bait}_upset_gprofiler_dotplot.pdf')
