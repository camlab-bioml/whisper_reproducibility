"""Sensitivity to how the GO:CC reference is built (Supplementary Fig. 8)."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.benchmark import (METHOD_COLORS, is_any_gene_in_gold, load_new_methods,
                           load_old_methods, paired_bootstrap_aucs, paired_difference_ci)

OUT_SUBDIR = 'gocc_sensitivity'
CAPS = [100, 250, 500, 1000, 5000]
IEA_SETTINGS = [True, False]          
EXCLUDE_TERMS = {
    'GO:0005575',  # cellular component
    'GO:0005623',  # cell
    'GO:0005576',  # extracellular region
    'GO:0005622',  # intracellular
    'GO:0043226',  # organelle
    'GO:0005624',  # membrane
}
GAF_COLS = ['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
            'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect',
            'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 'Taxon_ID',
            'Date', 'Assigned_By', 'Annotation_Extension', 'Gene_Product_Form_ID']
DEPTHS = {'Total AUC': None, 'AUC@Top300': 300}
DEPTH_LABELS = list(DEPTHS)
METHODS = ['WHISPER', 'Composite score', 'SAINTq', 'SAINTexpress', 'limma']
N_BOOTSTRAP = 100
OLD_RELEASE = '2021-01-01'
RELEASES = {'2024': '', OLD_RELEASE[:4]: f'_{OLD_RELEASE[:4]}'}


def cap_label(cap):
    return 'none' if cap is None else str(cap)


def config_label(cap, iea):
    return f'cap {cap_label(cap)} / IEA {"in" if iea else "out"}'


def load_gaf(path):
    df = pd.read_csv(path, sep='\t', comment='!', names=GAF_COLS, dtype=str)
    df = df[(df['Aspect'] == 'C') & (~df['GO_ID'].isin(EXCLUDE_TERMS))]
    return df[['DB_Object_Symbol', 'GO_ID', 'Evidence_Code', 'Qualifier']].dropna(
        subset=['DB_Object_Symbol', 'GO_ID'])


def gaf_members(gaf, dag):

    gaf = gaf[~gaf['Qualifier'].fillna('').str.contains('NOT')]
    members = {}
    for term, syms in gaf.groupby('GO_ID')['DB_Object_Symbol']:
        if term not in dag:
            continue
        for t in {term} | dag[term].get_all_parents():
            members.setdefault(t, set()).update(syms)
    return {t: sorted(s) for t, s in members.items()}


def bait_terms(gaf, dag, bait, include_iea):

    rows = gaf[gaf['DB_Object_Symbol'] == bait]
    if not include_iea:
        rows = rows[rows['Evidence_Code'] != 'IEA']
    terms = set(rows['GO_ID'].unique())
    expanded = set()
    for t in terms:
        if t in dag:
            expanded.add(t)
            expanded |= dag[t].get_all_parents()
    return {t for t in expanded if t not in EXCLUDE_TERMS}


def fetch_members(terms, cache, gp):

    missing = [t for t in sorted(terms) if t not in cache]
    if not missing:
        return []
    print(f'  fetching {len(missing)} term(s) from g:Profiler '
          f'({len(cache)} already cached)')
    failed = []
    for i, term in enumerate(missing, 1):
        try:
            res = gp.convert(organism='hsapiens', query=term,
                             target_namespace='ENTREZGENE')
            cache[term] = sorted(set(res['name'].dropna().unique())) \
                if not res.empty else []
        except Exception as exc:                      # noqa: BLE001
            failed.append((term, str(exc)[:60]))
        if i % 25 == 0:
            print(f'    {i}/{len(missing)}', flush=True)
    return failed


def gold_set(terms, cache, bait, cap):
    preys, kept = set(), 0
    for term in terms:
        members = set(cache.get(term, []))
        members.discard(bait)                 # self removed before the test
        if len(members) > 1 and (cap is None or len(members) <= cap):
            preys |= members
            kept += 1
    return preys, kept


def summary_plot(auc, out_pdf, dataset_name, deployed_cap):

    caps = [cap_label(c) for c in CAPS]
    methods = [m for m in METHODS if m in set(auc['Method'])]
    panels = [(depth, iea) for depth in DEPTH_LABELS for iea in IEA_SETTINGS]
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6.5),
                             squeeze=False)

    for ax, (depth, iea) in zip(axes[0], panels):
        s = auc[(auc['depth'] == depth) & (auc['IEA'] == iea)]
        if s.empty:
            ax.set_axis_off()
            continue
        sns.barplot(data=s, x='cap', y='AUC', hue='Method',
                    palette=METHOD_COLORS, ax=ax, order=caps,
                    hue_order=methods)
        for i, cap in enumerate(caps):
            sub = (s[s['cap'] == cap].set_index('Method')
                   .reindex(methods).reset_index())
            xs = [i - 0.4 + (j + 0.5) * (0.8 / len(sub))
                  for j in range(len(sub))]
            ax.errorbar(xs, sub['AUC'], yerr=sub['AUC_std'], fmt='none',
                        c='black', capsize=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel('AUCPR', fontsize=22)
        ax.set_xlabel('Term-size cap', fontsize=22)
        ax.set_title(f'{depth} — IEA {"included" if iea else "excluded"}',
                     fontsize=20)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        if ax.legend_:
            ax.legend_.remove()

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(dataset_name, fontsize=24)
    fig.legend(handles, labels, title='Method', ncol=len(methods),
               loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=15,
               title_fontsize=16, frameon=False)
    plt.tight_layout(rect=[0, 0.12, 1, 0.94])
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def label_specs(frames, gold):
    specs = []
    for m in METHODS:
        if m not in frames:
            continue
        df, prey_col, score_col = frames[m]
        d = df.copy()
        d['true_label'] = [is_any_gene_in_gold(p, gold.get(b, set()))
                           for b, p in zip(d['Bait'], d[prey_col])]
        specs.append((m, d, prey_col, score_col))
    return specs


def run_dataset(cfg, dag, gaf, cache, gp=None, suffix=''):

    outdir = os.path.join(cfg['outdir'], OUT_SUBDIR)
    os.makedirs(outdir, exist_ok=True)
    print(f'\n{"=" * 72}\n{cfg["name"]}\n{"=" * 72}')

    features_df, frames = load_new_methods(cfg, None)
    frames.update(load_old_methods(cfg, features_df))
    frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')
    identified = set(features_df['Prey'].astype(str))
    baits = list(features_df['Bait'].unique())
    deployed_cap = cfg['deployed_cap']

    # Every term either arm can ask about, fetched once.
    terms_by = {(b, iea): bait_terms(gaf, dag, b, iea)
                for b in baits for iea in IEA_SETTINGS}
    failed = (fetch_members(set().union(*terms_by.values()), cache, gp)
              if gp is not None else [])
    if failed:
        print(f'  WARNING: {len(failed)} term(s) failed to fetch and are '
              f'absent from every reference: {failed[:3]}')

    auc_rows, ref_rows, diff_rows = [], [], []
    for iea in IEA_SETTINGS:
        for cap in CAPS:
            name = config_label(cap, iea)
            gold, kept_total = {}, 0
            for b in baits:
                gold[b], kept = gold_set(terms_by[(b, iea)], cache, b, cap)
                gold[b] |= set(cfg['bait_self_prey'].get(b, [b]))
                # Restricted to the preys the run detected, as the main figures
                # restrict their reference (gt['Prey'].isin(identified)).
                gold[b] &= identified
                kept_total += kept

            specs = label_specs(frames, gold)

            n_pos = int(specs[0][1]['true_label'].sum())
            n_pairs = int(len(specs[0][1]))
            gold_pairs = sum(len(v & identified) for v in gold.values())
            ref_rows.append({'config': name, 'cap': cap_label(cap), 'IEA': iea,
                             'terms_kept': kept_total, 'gold_pairs': gold_pairs,
                             'labelled_positive': n_pos, 'pairs': n_pairs,
                             'base_rate': n_pos / n_pairs,
                             'is_deployed': (cap == deployed_cap and iea)})

            if n_pos == 0:
                print(f'  {name}: no annotated pairs, skipped')
                continue

            boot = paired_bootstrap_aucs(specs, DEPTHS,
                                         n_bootstrap=N_BOOTSTRAP, seed=0)
            frame_rows = []
            for m, _d, _pc, _sc in specs:
                row = {'Method': m}
                for depth in DEPTH_LABELS:
                    a = boot[m][depth]
                    row[depth] = float(a.mean())
                    row[f'{depth} Std'] = float(a.std())
                    auc_rows.append({'config': name, 'cap': cap_label(cap),
                                     'IEA': iea, 'Method': m, 'depth': depth,
                                     'AUC': float(a.mean()),
                                     'AUC_std': float(a.std())})
                frame_rows.append(row)
            for depth in DEPTH_LABELS:
                for other in [m for m in boot if m != 'WHISPER']:
                    r = paired_difference_ci(boot, 'WHISPER', other, depth)
                    r.update({'config': name, 'cap': cap_label(cap), 'IEA': iea,
                              'comparison': f'WHISPER - {other}',
                              'reference_method': other, 'depth': depth})
                    diff_rows.append(r)

            auc_df = pd.DataFrame(frame_rows)
            tag = f'cap{cap_label(cap)}_IEA{"in" if iea else "out"}'
            auc_df.to_csv(f'{outdir}/auc_source_data_{tag}{suffix}.csv',
                          index=False)

            best = auc_df.sort_values('AUC@Top300', ascending=False).iloc[0]
            print(f'  {name}: {kept_total} terms kept, base rate '
                  f'{100 * n_pos / n_pairs:.1f}%, top300 winner '
                  f'{best["Method"]} ({best["AUC@Top300"]:.4f})')

    auc = pd.DataFrame(auc_rows)
    auc.to_csv(f'{outdir}/auc_by_reference{suffix}.csv', index=False)
    pd.DataFrame(ref_rows).to_csv(f'{outdir}/reference_summary{suffix}.csv',
                                  index=False)
    pd.DataFrame(diff_rows).to_csv(
        f'{outdir}/auc_differences_by_reference{suffix}.csv', index=False)
    if not auc.empty and not suffix:
        summary_plot(auc, f'{outdir}/auc_gocc_sensitivity.pdf', cfg['name'],
                     deployed_cap)
    print(f'  wrote {outdir}')
    return auc.assign(dataset=cfg['name'])


def one_row_plot(auc, out_pdf, dataset_name):

    caps = [cap_label(c) for c in CAPS]
    blocks = [(rel, iea) for rel in RELEASES for iea in IEA_SETTINGS]
    order = [f'{rel}|{iea}|{c}' for rel, iea in blocks for c in caps]
    auc = auc.assign(x=auc['release'].astype(str) + '|' + auc['IEA'].astype(str)
                     + '|' + auc['cap'].astype(str))
    methods = [m for m in METHODS if m in set(auc['Method'])]
    fig, axes = plt.subplots(1, len(DEPTH_LABELS), squeeze=False,
                             figsize=(18 * len(DEPTH_LABELS), 7))

    for ax, depth in zip(axes[0], DEPTH_LABELS):
        s = auc[auc['depth'] == depth]
        sns.barplot(data=s, x='x', y='AUC', hue='Method',
                    palette=METHOD_COLORS, ax=ax, order=order,
                    hue_order=methods)
        for i, x in enumerate(order):
            sub = (s[s['x'] == x].set_index('Method')
                   .reindex(methods).reset_index())
            xs = [i - 0.4 + (j + 0.5) * (0.8 / len(sub))
                  for j in range(len(sub))]
            ax.errorbar(xs, sub['AUC'], yerr=sub['AUC_std'], fmt='none',
                        c='black', capsize=2)
        n = len(caps)
        for k, (rel, iea) in enumerate(blocks):
            if k:
                ax.axvline(k * n - 0.5, color='grey', ls='--', lw=1)
            ax.text(k * n + (n - 1) / 2, 1.02,
                    f'GO {rel}, IEA {"included" if iea else "excluded"}',
                    ha='center', va='bottom', fontsize=16,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([x.split('|')[2] for x in order])
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel('AUCPR', fontsize=22)
        ax.set_xlabel('Term-size cap', fontsize=22)
        ax.set_title(depth, fontsize=22, pad=34)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=18)
        if ax.legend_:
            ax.legend_.remove()

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(dataset_name, fontsize=24)
    fig.legend(handles, labels, title='Method', ncol=len(methods),
               loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=15,
               title_fontsize=16, frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 0.94])
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def run(cfg, gaf_file, obo_file, cache_file, old_release_dir):

    from goatools.obo_parser import GODag
    with open(cache_file) as fh:
        cache = json.load(fh)
    gp = None
    try:
        from gprofiler import GProfiler
        gp = GProfiler(return_dataframe=True)
    except ImportError:
        pass
    parts = [run_dataset(cfg, GODag(obo_file), load_gaf(gaf_file), cache, gp).assign(release='2024')]
    with open(cache_file, 'w') as fh:
        json.dump(cache, fh)

    dag = GODag(os.path.join(old_release_dir, 'go-basic.obo'))
    gaf = load_gaf(os.path.join(old_release_dir, 'goa_human.gaf'))
    old = run_dataset(cfg, dag, gaf, gaf_members(gaf, dag), suffix=RELEASES[OLD_RELEASE[:4]])
    parts.append(old.assign(release=OLD_RELEASE[:4]))
    auc = pd.concat(parts, ignore_index=True)
    auc['cap'] = auc['cap'].astype(str)
    one_row_plot(auc, os.path.join(cfg['outdir'], OUT_SUBDIR, 'auc_gocc_sensitivity_one_row.pdf'),
                 cfg['name'])
