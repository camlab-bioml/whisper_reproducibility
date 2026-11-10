"""Benchmark against references built from each GO aspect
(Supplementary Fig. 7)."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.benchmark import (METHOD_COLORS, is_any_gene_in_gold, load_new_methods,
                           load_old_methods, paired_bootstrap_aucs, paired_difference_ci)

ASPECTS = {'CC': 'C', 'BP': 'P', 'MF': 'F', 'ALL': None}
OUT_SUBDIR = 'go_aspects'
DEPTHS = {'Total AUC': None, 'AUC@Top300': 300}
DEPTH_LABELS = list(DEPTHS)
METHODS = ['WHISPER', 'Composite score', 'SAINTq', 'SAINTexpress', 'limma',
           'CompPASS', 'SFINX', 'PP-NSAF', 'MiST']
MAIN_METHODS = ['WHISPER', 'Composite score', 'SAINTq', 'SAINTexpress', 'limma']
N_BOOTSTRAP = 100

GAF_COLS = ['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
            'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect',
            'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 'Taxon_ID',
            'Date', 'Assigned_By', 'Annotation_Extension', 'Gene_Product_Form_ID']
EXCLUDE_TERMS = {
    'GO:0005575',  # cellular component (CC root)
    'GO:0005623',  # cell
    'GO:0005576',  # extracellular region
    'GO:0005622',  # intracellular
    'GO:0043226',  # organelle
    'GO:0005624',  # membrane
    'GO:0008150',  # biological process (BP root)
    'GO:0003674',  # molecular function (MF root)
}


# ---------------------------------------------------------------- references

def bait_terms(gaf, dag, bait, aspect_code):
    """A bait's terms for one aspect, expanded to ancestors."""
    rows = gaf[gaf['DB_Object_Symbol'] == bait]
    if aspect_code is not None:
        rows = rows[rows['Aspect'] == aspect_code]
    expanded = set()
    for t in set(rows['GO_ID'].unique()):
        if t in dag:
            expanded.add(t)
            expanded |= dag[t].get_all_parents()
    return {t for t in expanded if t not in EXCLUDE_TERMS}


def fetch_members(terms, cache, gp):
    """term -> member gene symbols from g:Profiler, cached on disk.

    A failed lookup is reported rather than cached as an empty set, which
    would be indistinguishable from a term that genuinely has no members.
    """
    missing = [t for t in sorted(terms) if t not in cache]
    if not missing:
        return []
    print(f'  fetching {len(missing)} term(s) from g:Profiler '
          f'({len(cache)} cached)')
    failed = []
    for i, term in enumerate(missing, 1):
        try:
            res = gp.convert(organism='hsapiens', query=term,
                             target_namespace='ENTREZGENE')
            cache[term] = (sorted(set(res['name'].dropna().unique()))
                           if not res.empty else [])
        except Exception as exc:                      # noqa: BLE001
            failed.append((term, str(exc)[:60]))
        if i % 50 == 0:
            print(f'    {i}/{len(missing)}', flush=True)
    return failed


def gold_set(terms, cache, bait, cap):
    preys, kept = set(), 0
    for term in terms:
        members = set(cache.get(term, []))
        members.discard(bait)                 # self removed before the size test
        if 1 < len(members) <= cap:
            preys |= members
            kept += 1
    return preys, kept


def generate_references(baits, cap, outdir, gaf_file, obo_file, cache_file):
    """Write go_<aspect>_interactions.csv and reference_summary.csv to outdir.
    Terms missing from the cache are fetched from g:Profiler and added to it."""
    from goatools.obo_parser import GODag
    os.makedirs(outdir, exist_ok=True)
    dag = GODag(obo_file)
    gaf = pd.read_csv(gaf_file, sep='\t', comment='!', names=GAF_COLS, dtype=str)
    gaf = gaf[~gaf['GO_ID'].isin(EXCLUDE_TERMS)][['DB_Object_Symbol', 'GO_ID', 'Aspect']].dropna()
    with open(cache_file) as fh:
        cache = json.load(fh)
    terms_by = {(b, a): bait_terms(gaf, dag, b, code) for b in baits for a, code in ASPECTS.items()}
    missing = set().union(*terms_by.values()) - set(cache)
    if missing:
        from gprofiler import GProfiler
        failed = fetch_members(missing, cache, GProfiler(return_dataframe=True))
        with open(cache_file, 'w') as fh:
            json.dump(cache, fh)
        if failed:
            print(f'WARNING: {len(failed)} term(s) failed to fetch: {failed[:3]}')
    rows = []
    for aspect in ASPECTS:
        frames = []
        for b in baits:
            preys, kept = gold_set(terms_by[(b, aspect)], cache, b, cap)
            frames.append(pd.DataFrame({'Bait': b, 'Prey': sorted(preys)}))
            rows.append({'aspect': aspect, 'Bait': b, 'terms': len(terms_by[(b, aspect)]),
                         'terms_kept': kept, 'preys': len(preys)})
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(outdir, f'go_{aspect}_interactions.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'reference_summary.csv'), index=False)


# ---------------------------------------------------------------- benchmark

def figure(auc, out_pdf, dataset_name, only=None):
    """One panel per aspect; Total AUC and AUC@Top300 within each. `only`
    restricts the bars to those methods (same values, fewer bars)."""
    aspects = [a for a in ASPECTS if a in set(auc['aspect'])]
    methods = [m for m in (only or METHODS) if m in set(auc['Method'])]
    auc = auc[auc['Method'].isin(methods)]
    fig, axes = plt.subplots(1, len(aspects), figsize=(6.5 * len(aspects), 7),
                             squeeze=False)
    for ax, aspect in zip(axes[0], aspects):
        s = auc[auc['aspect'] == aspect]
        melted = s.melt(id_vars='Method', value_vars=DEPTH_LABELS,
                        var_name='AUC Type', value_name='AUC')
        melted['Std'] = s.melt(id_vars='Method',
                               value_vars=[f'{d} Std' for d in DEPTH_LABELS],
                               var_name='AUC Type', value_name='Std')['Std']
        sns.barplot(data=melted, x='AUC Type', y='AUC', hue='Method',
                    palette=METHOD_COLORS, ax=ax, order=DEPTH_LABELS,
                    hue_order=methods)
        for i, depth in enumerate(DEPTH_LABELS):
            sub = (melted[melted['AUC Type'] == depth].set_index('Method')
                   .reindex(methods).reset_index())
            xs = [i - 0.4 + (j + 0.5) * (0.8 / len(sub)) for j in range(len(sub))]
            ax.errorbar(xs, sub['AUC'], yerr=sub['Std'], fmt='none', c='black',
                        capsize=3)
        ax.set_ylabel('AUC Value', fontsize=22)
        # Fixed 0-1, as the main AUC bar plots, so panels line up.
        ax.set_ylim(0, 1)
        ax.set_xlabel('', fontsize=22)
        ax.set_title(f'GO:{aspect}', fontsize=20)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=18)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(dataset_name, fontsize=24)
    fig.legend(handles, labels, title='Method', ncol=len(methods),
               loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=15,
               title_fontsize=16, frameon=False)
    plt.tight_layout(rect=[0, 0.14, 1, 0.95])
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def run_dataset(cfg, reference_dir):
    """cfg: the dictionary src.benchmark.run_dataset takes; reference_dir holds
    go_<CC|BP|MF|ALL>_interactions.csv."""
    outdir = os.path.join(cfg['outdir'], OUT_SUBDIR)
    os.makedirs(outdir, exist_ok=True)
    print(f'\n{"=" * 72}\n{cfg["name"]}\n{"=" * 72}')

    features_df, frames = load_new_methods(cfg, None)
    frames.update(load_old_methods(cfg, features_df))
    frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')
    identified = set(features_df['Prey'].astype(str))
    baits = list(features_df['Bait'].unique())

    auc_rows, diff_rows = [], []
    for aspect in ASPECTS:
        path = os.path.join(reference_dir, f'go_{aspect}_interactions.csv')
        if not os.path.exists(path):
            print(f'  {aspect}: no reference at {path} -- skipped')
            continue
        gt = pd.read_csv(path)
        gt = gt[gt['Prey'].isin(identified)]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}

        specs = []
        for m in METHODS:
            if m not in frames:
                continue
            df, prey_col, score_col = frames[m]
            d = df.copy()
            d['true_label'] = [is_any_gene_in_gold(p, gold.get(b, set()))
                               for b, p in zip(d['Bait'], d[prey_col])]
            specs.append((m, d, prey_col, score_col))

        n_pos, n_pairs = int(specs[0][1]['true_label'].sum()), len(specs[0][1])
        print(f'  GO:{aspect:<4} {n_pos}/{n_pairs} pairs annotated')
        if n_pos == 0:
            continue

        boot = paired_bootstrap_aucs(specs, DEPTHS, n_bootstrap=N_BOOTSTRAP,
                                     seed=0)
        for m, _d, _p, _s in specs:
            row = {'aspect': aspect, 'Method': m}
            for depth in DEPTH_LABELS:
                a = boot[m][depth]
                row[depth] = float(a.mean())
                row[f'{depth} Std'] = float(a.std())
            auc_rows.append(row)
        for depth in DEPTH_LABELS:
            for other in [m for m in boot if m != 'WHISPER']:
                r = paired_difference_ci(boot, 'WHISPER', other, depth)
                r.update({'aspect': aspect, 'comparison': f'WHISPER - {other}',
                          'reference_method': other, 'depth': depth})
                diff_rows.append(r)

    auc = pd.DataFrame(auc_rows)
    auc.to_csv(os.path.join(outdir, 'auc_by_aspect.csv'), index=False)
    pd.DataFrame(diff_rows).to_csv(
        os.path.join(outdir, 'auc_differences_by_aspect.csv'), index=False)
    if not auc.empty:
        figure(auc, os.path.join(outdir, 'auc_go_aspects.pdf'), cfg['name'])
        figure(auc, os.path.join(outdir, 'auc_go_aspects_main_methods.pdf'),
               cfg['name'], only=MAIN_METHODS)
    print(f'  wrote {outdir}')
