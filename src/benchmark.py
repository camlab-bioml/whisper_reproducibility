"""Benchmark of WHISPER against the other methods (Fig. 2, Fig. 3,
Supplementary Figs. 9 and 10)."""

import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# =============================================================================
# Config
# =============================================================================

STEP_SIZE = 10
GAF_FILE = 'src/goa_human.gaf'

FEATURE_NAMES = [
    'log_fold_change', 'snr', 'mean_diff', 'median_diff',
    'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
    'zero_or_neg_fc',
]


DEPTHS = {'Total AUC': None, 'AUC@Top300': 300}
DEPTH_LABELS = list(DEPTHS)
N_BOOTSTRAP = 100


METHOD_COLORS = {
    'WHISPER': '#003f5c',
    'PU Learning': '#003f5c',
    'SAINTq': '#ffb347',
    'SAINTexpress': '#ff7f0e',
    'Composite': '#7a5195',
    'Composite score': '#7a5195',
    'limma': '#d62728',
}
METHOD_COLORS.update(dict(zip(
    FEATURE_NAMES, sns.color_palette('YlGnBu', n_colors=len(FEATURE_NAMES)))))

METHOD_COLORS['CompPASS'] = '#2a9d8f'
METHOD_COLORS['SFINX'] = '#9c755f'
METHOD_COLORS['PP-NSAF'] = '#59a14f'
METHOD_COLORS['MiST'] = '#e377c2'


TAG_PREFIX = 'all'


OLD_METHODS = ['WHISPER', 'Composite score', 'SAINTq', 'SAINTexpress', 'limma']
NEW_METHODS = ['CompPASS', 'SFINX', 'PP-NSAF', 'MiST']
ALL_METHODS = OLD_METHODS + NEW_METHODS


MAIN_METHODS = OLD_METHODS


def paired_bootstrap_aucs_by_bait(specs, depths, baits,
                                  n_bootstrap=N_BOOTSTRAP, seed=0):
    """As paired_bootstrap_aucs, but the resampling unit is the BAIT.

    Baits are drawn with replacement and each carries all of its pairs intact,
    which respects the clustering the pair-level bootstrap ignores. With a
    handful of baits per dataset the resulting intervals are wide; that width
    is the honest reflection of how few independent experiments there are.
    """
    keys = [list(zip(df['Bait'].astype(str), df[prey_col].astype(str)))
            for _n, df, prey_col, _sc in specs]
    common = set(keys[0])
    for k in keys[1:]:
        common &= set(k)
    seen, order = set(), []
    for k in keys[0]:
        if k in common and k not in seen:
            seen.add(k)
            order.append(k)

    aligned = {}
    for (name, df, prey_col, score_col), k in zip(specs, keys):
        d = df.copy()
        d['_pair_key'] = k
        d = d.drop_duplicates('_pair_key').set_index('_pair_key').reindex(order)
        score = pd.to_numeric(d[score_col], errors='coerce').to_numpy(dtype=float)
        aligned[name] = (np.where(np.isnan(score), -np.inf, score),
                         d['true_label'].to_numpy(dtype=float))

    bait_of = np.array([k[0] for k in order])
    bait_rows = {b: np.flatnonzero(bait_of == str(b)) for b in baits}
    bait_list = [b for b in baits if len(bait_rows[b])]

    rng = np.random.default_rng(seed)
    out = {name: {lab: np.empty(n_bootstrap) for lab in depths} for name in aligned}
    for b in range(n_bootstrap):
        picked = rng.choice(len(bait_list), len(bait_list), replace=True)
        idx = np.concatenate([bait_rows[bait_list[i]] for i in picked])
        for name, (score, true) in aligned.items():
            t_sorted = tie_averaged_labels(score[idx], true[idx])
            for lab, topn in depths.items():
                out[name][lab][b] = _pr_auc_from_sorted(
                    t_sorted if topn is None else t_sorted[:topn])
    return out


# =============================================================================
# Shared helpers -- transcribed from plot_precision_overall_rank_bait.py
# =============================================================================

def add_missing_preys(df, features_df, prey_col):

    updated_df = pd.DataFrame()
    for bait in features_df['Bait'].unique():
        unique_preys = set(features_df[features_df['Bait'] == bait]['Prey'])
        existing_preys = set(df[df['Bait'] == bait][prey_col])
        missing_preys = unique_preys - existing_preys
        if missing_preys:
            print(f"Adding {len(missing_preys)} missing preys for bait {bait}.")
            missing_df = pd.DataFrame({
                'Bait': [bait] * len(missing_preys),
                prey_col: list(missing_preys),
                'BFDR': [1.0] * len(missing_preys),
                'AvgP': [0] * len(missing_preys)
            })
            updated_df = pd.concat([updated_df, missing_df], ignore_index=True)
        updated_df = pd.concat([updated_df, df[df['Bait'] == bait]],
                               ignore_index=True)
    return updated_df


def load_uniprot_to_gene(path=GAF_FILE):

    mapping = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) > 2 and fields[1] and fields[2]:
                mapping.setdefault(fields[1], fields[2])
    return mapping


def make_to_gene(id_map):

    def to_gene(value):
        for member in str(value).split(';'):
            accession = member.split('-')[0].strip()
            if accession in id_map:
                return id_map[accession]
        return value
    return to_gene


def tie_averaged_labels(score, true):

    score = np.asarray(score, dtype=float)
    score = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-score, kind='stable')
    s, t = score[order], np.asarray(true, dtype=float)[order]
    if len(s) == 0:
        return t
    block = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    return (np.bincount(block, weights=t) / np.bincount(block))[block]


def calculate_precision_recall_f1_topn(df, score_column, true_label_column,
                                       step_size=STEP_SIZE):
    # Ties take their block's mean hit rate, as in the bootstrap AUCs.
    cum = np.cumsum(tie_averaged_labels(
        pd.to_numeric(df[score_column], errors='coerce').to_numpy(dtype=float),
        df[true_label_column].to_numpy(dtype=float)))
    precision, recall, f1_scores, threshold_dots = [], [], [], {}

    n_total_positives = cum[-1] if len(cum) else 0
    if n_total_positives == 0:
        print("Warning: no positive labels found!")

    topn_list = np.arange(step_size, len(cum) + step_size, step_size)
    for topn in topn_list:
        tp = cum[min(topn, len(cum)) - 1]
        fp = topn - tp
        fn = n_total_positives - tp

        precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_value = ((2 * precision_value * recall_value)
                    / (precision_value + recall_value)
                    if (precision_value + recall_value) > 0 else 0)

        precision.append(precision_value)
        recall.append(recall_value)
        f1_scores.append(f1_value)

        if topn in [100, 300]:
            threshold_dots[topn] = (recall_value, precision_value)

    return np.array(recall), np.array(precision), f1_scores, threshold_dots


def _pr_auc_from_sorted(true_sorted, step_size=STEP_SIZE):

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


def paired_bootstrap_aucs(specs, depths, n_bootstrap=N_BOOTSTRAP, seed=0):

    keys = [list(zip(df['Bait'].astype(str), df[prey_col].astype(str)))
            for _n, df, prey_col, _sc in specs]

    common = set(keys[0])
    for k in keys[1:]:
        common &= set(k)
    seen, order = set(), []
    for k in keys[0]:
        if k in common and k not in seen:
            seen.add(k)
            order.append(k)

    aligned = {}
    for (name, df, prey_col, score_col), k in zip(specs, keys):
        d = df.copy()
        d['_pair_key'] = k
        d = d.drop_duplicates('_pair_key').set_index('_pair_key').reindex(order)
        score = pd.to_numeric(d[score_col], errors='coerce').to_numpy(dtype=float)

        score = np.where(np.isnan(score), -np.inf, score)
        aligned[name] = (score, d['true_label'].to_numpy(dtype=float))

    rng = np.random.default_rng(seed)
    n = len(order)
    out = {name: {lab: np.empty(n_bootstrap) for lab in depths} for name in aligned}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)          # one draw, shared by every method
        for name, (score, true) in aligned.items():
            t_sorted = tie_averaged_labels(score[idx], true[idx])
            for lab, topn in depths.items():
                out[name][lab][b] = _pr_auc_from_sorted(
                    t_sorted if topn is None else t_sorted[:topn])
    return out


def paired_difference_ci(boot, a, b, label, alpha=0.05):
    d = boot[a][label] - boot[b][label]
    return {'mean_difference': float(d.mean()),
            'ci_low': float(np.percentile(d, 100 * alpha / 2)),
            'ci_high': float(np.percentile(d, 100 * (1 - alpha / 2))),
            'frac_replicates_positive': float((d > 0).mean())}


def is_any_gene_in_gold(prey_str, gold_set):
    if pd.isna(prey_str):
        return 0
    return int(any(gene in gold_set for gene in str(prey_str).split(';')))



# =============================================================================
# Inputs
# =============================================================================

def load_new_methods(cfg, key):
    features_df = pd.read_csv(cfg['features'])

    # CompPASS, from cRomppass -- see comppass_run.R
    cp_path = cfg['comppass']
    comppass = pd.read_csv(cp_path)[['Bait', 'Prey', 'WD', 'Z', 'Entropy']]


    to_gene = make_to_gene(load_uniprot_to_gene(cfg['gaf_file'])) if cfg['gaf'] else None
    if cfg['gaf']:
        for label, frame, col in [('WHISPER', features_df, 'Prey'),
                                  ('CompPASS', comppass, 'Prey')]:
            if frame is None:
                continue
            before = frame[col].astype(str)
            frame[col] = before.map(to_gene)
            print(f'{label}: {int((frame[col].astype(str) != before).sum())}/'
                  f'{len(frame)} identifiers converted to gene symbols')

    sx_path = cfg['sfinx']
    sfinx = pd.read_csv(sx_path)
    if not sfinx.empty:

        back = {v: k for k, v in cfg.get('self_gene', {}).items()}
        back.update({'P25789': 'PSMA4', 'P35998': 'PSMC2',
                     'ACTB;ACTG1': 'ACTB', 'NRAS;HRAS': 'NRAS',
                     'RAC1;RAC3;RAC2': 'RAC1', 'RHOA;RHOC': 'RHOA'})
        sfinx['Bait'] = sfinx['Bait'].map(lambda b: back.get(b, b))
        if to_gene is not None:
            sfinx['Prey'] = sfinx['Prey'].astype(str).map(to_gene)
    pairs = features_df[['Bait', 'Prey']].drop_duplicates().copy()
    comppass = pairs.merge(comppass, on=['Bait', 'Prey'], how='left')
    sfinx = pairs.merge(
        sfinx[['Bait', 'Prey', 'sfinx_score']] if not sfinx.empty
        else pd.DataFrame(columns=['Bait', 'Prey', 'sfinx_score']),
        on=['Bait', 'Prey'], how='left')
    n_scored = int(sfinx['sfinx_score'].notna().sum())
    sfinx['sfinx_score'] = sfinx['sfinx_score'].fillna(0.0)
    print(f'SFINX: {n_scored}/{len(sfinx)} pairs retained and scored; '
          f'the rest padded to the bottom of the ranking')

    pp_path = cfg['ppnsaf']
    ppnsaf = pd.read_csv(pp_path)[['Bait', 'Prey', 'ppnsaf_score']]
    if to_gene is not None:
        back_pp = {v: k for k, v in cfg.get('self_gene', {}).items()}
        back_pp.update({'P25789': 'PSMA4', 'P35998': 'PSMC2'})
        ppnsaf['Bait'] = ppnsaf['Bait'].map(lambda b: back_pp.get(b, b))
        ppnsaf['Prey'] = ppnsaf['Prey'].astype(str).map(to_gene)
    ppnsaf = pairs.merge(ppnsaf, on=['Bait', 'Prey'], how='left')
    n_pp = int(ppnsaf['ppnsaf_score'].notna().sum())
    ppnsaf['ppnsaf_score'] = ppnsaf['ppnsaf_score'].fillna(0.0)
    print(f'PP-NSAF: {n_pp}/{len(ppnsaf)} pairs surviving the contaminant '
          f'filter and scored; the rest padded to the bottom')

    mist = None
    mi_path = cfg.get('mist')
    if mi_path and os.path.exists(mi_path):
        mist = pd.read_csv(mi_path)[['Bait', 'Prey', 'mist_score']]
        if to_gene is not None:
            mist['Prey'] = mist['Prey'].astype(str).map(to_gene)
        mist = pairs.merge(mist, on=['Bait', 'Prey'], how='left')
        n_mi = int(mist['mist_score'].notna().sum())
        mist['mist_score'] = mist['mist_score'].fillna(-1.0)
        print(f'MiST: {n_mi}/{len(mist)} pairs scored; the rest padded to '
              f'the bottom')
    else:
        print(f'MiST: no scores at {mi_path} -- omitted for this dataset')


    frames = {'WHISPER': (features_df, 'Prey', 'predicted_probability'),
              'CompPASS': (comppass, 'Prey', 'WD'),
              'SFINX': (sfinx, 'Prey', 'sfinx_score'),
              'PP-NSAF': (ppnsaf, 'Prey', 'ppnsaf_score')}
    if mist is not None:
        frames['MiST'] = (mist, 'Prey', 'mist_score')
    return features_df, frames


def load_old_methods(cfg, features_df):

    saintq = saintex = limma = None
    if cfg['saintq']:
        saintq = pd.read_csv(cfg['saintq'], sep='\t').rename(
            columns={'Prey': 'PreyGene'})
    if cfg['saintexpress']:
        saintex = pd.read_csv(cfg['saintexpress'], sep='\t')
    if cfg['limma']:
        limma = pd.read_csv(cfg['limma'], sep='\t').rename(
            columns={'Protein': 'Prey'})
        if cfg['limma_positive_only']:
            limma = limma[limma['logFC'] > 0]

    if cfg['gaf']:
        to_gene = make_to_gene(load_uniprot_to_gene(cfg['gaf_file']))
        for label, frame, col in [('SAINTq', saintq, 'PreyGene'),
                                  ('SAINTexpress', saintex, 'PreyGene'),
                                  ('limma', limma, 'Prey')]:
            if frame is None:
                continue
            before = frame[col].astype(str)
            frame[col] = before.map(to_gene)
            changed = int((frame[col].astype(str) != before).sum())
            print(f'{label}: {changed}/{len(frame)} identifiers converted to '
                  f'gene symbols ({len(frame) - changed} already symbols or '
                  f'unmapped)')


    if saintq is not None:
        saintq = add_missing_preys(saintq, features_df, 'PreyGene')
    if saintex is not None:
        saintex = add_missing_preys(saintex, features_df, 'PreyGene')

    if limma is not None:

        limma = (features_df[['Bait', 'Prey']].drop_duplicates()
                 .merge(limma, on=['Bait', 'Prey'], how='left'))
        limma['limma_score'] = limma['t']

    features_df['composite_score_norm'] = MinMaxScaler().fit_transform(
        features_df[['composite_score']]).flatten()

    frames = {'Composite score': (features_df, 'Prey', 'composite_score_norm')}
    if saintq is not None:
        frames['SAINTq'] = (saintq, 'PreyGene', 'AvgP')
    if saintex is not None:
        frames['SAINTexpress'] = (saintex, 'PreyGene', 'AvgP')
    if limma is not None:
        frames['limma'] = (limma, 'Prey', 'limma_score')
    return frames


# =============================================================================
# Figures -- the layouts of plot_precision_overall_rank.py
# =============================================================================

def pr_plot(entries, out_pdf, title, features=None):
    plt.figure(figsize=(10, 10) if features else (8, 8))
    for label, recall, precision, m_auc, s_auc in entries:
        plt.plot(recall, precision,
                 label=f'{label} (AUC: {m_auc:.4f} ± {s_auc:.4f})',
                 marker='o', markersize=2, color=METHOD_COLORS.get(label))
    for label, recall, precision, m_auc, s_auc in (features or []):
        plt.plot(recall, precision, linestyle='--', linewidth=1,
                 color=METHOD_COLORS.get(label),
                 label=f'{label} (AUC: {m_auc:.4f} ± {s_auc:.4f})')
    plt.xlabel('Recall', fontsize=20 if features else 26)
    plt.ylabel('Precision', fontsize=20 if features else 26)
    plt.title(title, fontsize=18 if features else 26)
    plt.xticks(fontsize=16 if features else 24)
    plt.yticks(fontsize=16 if features else 24)
    plt.legend(loc='best', fontsize=8 if features else 14)
    if features:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def auc_bar(auc_df, out_pdf, ncol, anchor, title=''):
    melted = auc_df.melt(id_vars='Method', value_vars=DEPTH_LABELS,
                         var_name='AUC Type', value_name='AUC')
    melted['Std'] = auc_df.melt(
        id_vars='Method', value_vars=[f'{d} Std' for d in DEPTH_LABELS],
        var_name='AUC Type', value_name='Std')['Std']

    plt.figure(figsize=(10, 8))
    sns.barplot(data=melted, x='AUC Type', y='AUC', hue='Method',
                palette=METHOD_COLORS)
    for i, depth in enumerate(DEPTH_LABELS):
        sub = melted[melted['AUC Type'] == depth]
        xs = [i - 0.4 + (j + 0.5) * (0.8 / len(sub)) for j in range(len(sub))]
        plt.errorbar(xs, sub['AUC'], yerr=sub['Std'], fmt='none', c='black',
                     capsize=3)
    plt.ylabel('AUC Value', fontsize=26)

    plt.ylim(0, 1)
    plt.xlabel('', fontsize=26)
    if title:
        plt.title(title, fontsize=18)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(title='Method', ncol=ncol, loc='lower center',
               bbox_to_anchor=anchor, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


# =============================================================================
# Run
# =============================================================================

def run_dataset(cfg, key=None):
    """cfg: name, features, saintq, saintexpress, limma, comppass, sfinx,
    ppnsaf, mist, gts [(source, file)], outdir, self_gene, gaf, gaf_file,
    limma_positive_only, all_methods_in_main."""
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    print(f'\n{"=" * 72}\n{cfg["name"]}\n{"=" * 72}')


    features_df, frames = load_new_methods(cfg, key)
    frames.update(load_old_methods(cfg, features_df))
    frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')

    baits = list(features_df['Bait'].unique())
    identified = set(features_df['Prey'].astype(str))
    present = [m for m in ALL_METHODS if m in frames]
    print(f'  methods present: {", ".join(present)}')

    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(identified)]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}

        specs, curves = [], {}
        for m in present:
            df, prey_col, score_col = frames[m]
            d = df.copy()
            d['true_label'] = [is_any_gene_in_gold(p, gold.get(b, set()))
                               for b, p in zip(d['Bait'], d[prey_col])]
            specs.append((m, d, prey_col, score_col))
            r, p, f1, _ = calculate_precision_recall_f1_topn(
                d, score_col, 'true_label')
            curves[m] = (r, p, f1)

        whisper_df = specs[[s[0] for s in specs].index('WHISPER')][1]
        feature_names = [f for f in FEATURE_NAMES if f in whisper_df.columns]
        for feat in feature_names:
            specs.append((feat, whisper_df, 'Prey', feat))

        n_pos = int(whisper_df['true_label'].sum())
        print(f'  [{gt_source}] {n_pos}/{len(whisper_df)} pairs annotated '
              f'({100 * n_pos / len(whisper_df):.1f}%)')

        boot = paired_bootstrap_aucs(specs, DEPTHS, n_bootstrap=N_BOOTSTRAP,
                                     seed=0)
        boot_bait = paired_bootstrap_aucs_by_bait(specs, DEPTHS, baits,
                                                  n_bootstrap=N_BOOTSTRAP,
                                                  seed=0)

        def stats(name, label='Total AUC'):
            a = boot[name][label]
            return float(a.mean()), float(a.std())

        def auc_frame(methods):
            rows = []
            for m in methods:
                if m not in boot:
                    continue
                tot, tot_sd = stats(m, 'Total AUC')
                top, top_sd = stats(m, 'AUC@Top300')
                rows.append({'Method': m, 'Total AUC': tot,
                             'Total AUC Std': tot_sd, 'AUC@Top300': top,
                             'AUC@Top300 Std': top_sd})
            return pd.DataFrame(rows)

        def entries(methods):
            out = []
            for m in methods:
                if m in curves and m in boot:
                    r, p, _f1 = curves[m]
                    mu, sd = stats(m)
                    out.append((m, r, p, mu, sd))
            return out

        tag = f'{TAG_PREFIX}_{gt_source}'

        # --- paired differences, both resampling units ---------------------
        diff_rows = []
        for unit, bt in [('pair', boot), ('bait', boot_bait)]:
            for lab in DEPTHS:
                for other in [m for m in present if m != 'WHISPER']:
                    row = paired_difference_ci(bt, 'WHISPER', other, lab)
                    row.update({'comparison': f'WHISPER - {other}',
                                'reference': other, 'depth': lab,
                                'resampling_unit': unit,
                                'n_bootstrap': N_BOOTSTRAP,
                                'ground_truth': gt_source})
                    diff_rows.append(row)
        pd.DataFrame(diff_rows)[
            ['comparison', 'reference', 'depth', 'resampling_unit',
             'mean_difference', 'ci_low', 'ci_high',
             'frac_replicates_positive', 'n_bootstrap', 'ground_truth']
        ].to_csv(f'{outdir}/auc_paired_differences_{tag}.csv', index=False)

        # --- curve source data ---------------------------------------------
        rows = []
        for m, (r, p, f1) in curves.items():
            mu, sd = stats(m)
            rows.append(pd.DataFrame({'Method': m, 'Recall': r, 'Precision': p,
                                      'F1': f1, 'AUC': mu, 'AUC_std': sd,
                                      'ground_truth': gt_source}))
        pd.concat(rows, ignore_index=True).to_csv(
            f'{outdir}/precision_recall_source_data_{tag}.csv', index=False)

        # --- the method sets -----------------------------------------------

        main = (present if cfg.get('all_methods_in_main')
                else [m for m in present if m in MAIN_METHODS])
        variants = {
            'main': main,
            'main_no_heuristic': [m for m in main if m != 'Composite score'],
            'all_methods': present,
            'all_methods_no_heuristic': [m for m in present
                                         if m != 'Composite score'],
            'whisper_vs_heuristic': ['WHISPER', 'Composite score'],
        }
        for suffix, methods in variants.items():
            auc_df = auc_frame(methods)
            if suffix != 'whisper_vs_heuristic':
                pr_plot(entries(methods),
                        f'{outdir}/overall_precision_vs_recall_{tag}_{suffix}.pdf',
                        'Overall Precision vs. Recall (All Baits Combined)')
            auc_bar(auc_df, f'{outdir}/auc_barplot_{tag}_{suffix}.pdf',
                    4, (0.5, -0.32), title=cfg['name'])
            out = auc_df.copy()
            out['ground_truth'] = gt_source
            out.to_csv(f'{outdir}/auc_source_data_{tag}_{suffix}.csv',
                       index=False)
        auc_df_every = auc_frame(present)

        # --- supplementary: every method with the individual features -----
        feature_entries, feature_rows = [], []
        for feat in feature_names:
            fdf = whisper_df.dropna(subset=[feat, 'true_label'])
            if fdf.empty or feat not in boot:
                continue
            r, p, _f1, _ = calculate_precision_recall_f1_topn(
                fdf, feat, 'true_label')
            mu, sd = stats(feat)
            top, top_sd = stats(feat, 'AUC@Top300')
            feature_entries.append((feat, r, p, mu, sd))
            feature_rows.append({'Method': feat, 'Total AUC': mu,
                                 'Total AUC Std': sd, 'AUC@Top300': top,
                                 'AUC@Top300 Std': top_sd})

        pr_plot(entries(present),
                f'{outdir}/overall_precision_vs_recall_{tag}_with_features.pdf',
                'Including Individual Features', features=feature_entries)
        auc_df_features = pd.concat(
            [auc_df_every, pd.DataFrame(feature_rows)], ignore_index=True)
        auc_bar(auc_df_features,
                f'{outdir}/auc_barplot_{tag}_with_features.pdf',
                4, (0.5, -0.42), title=cfg['name'])
        out_feat = auc_df_features.copy()
        out_feat['ground_truth'] = gt_source
        out_feat.to_csv(f'{outdir}/auc_source_data_{tag}_with_features.csv',
                        index=False)

        # --- WHISPER against its own individual features only --------------

        auc_df_wf = pd.concat([auc_frame(['WHISPER']),
                               pd.DataFrame(feature_rows)], ignore_index=True)
        pr_plot(entries(['WHISPER']),
                f'{outdir}/overall_precision_vs_recall_{tag}_whisper_features.pdf',
                'WHISPER and Its Individual Features',
                features=feature_entries)
        auc_bar(auc_df_wf,
                f'{outdir}/auc_barplot_{tag}_whisper_features.pdf',
                3, (0.5, -0.42), title=cfg['name'])
        out_wf = auc_df_wf.copy()
        out_wf['ground_truth'] = gt_source
        out_wf.to_csv(f'{outdir}/auc_source_data_{tag}_whisper_features.csv',
                      index=False)

        order = auc_df_every.sort_values('AUC@Top300', ascending=False)
        print(f'  [{gt_source}] Top300 ranking: ' + ', '.join(
            f'{m} {v:.3f}' for m, v in zip(order['Method'],
                                           order['AUC@Top300'])))
    print(f'  wrote {outdir}')
