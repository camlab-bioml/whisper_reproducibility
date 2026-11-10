"""Benchmark bait by bait (Supplementary Fig. 5)."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.benchmark import (ALL_METHODS, FEATURE_NAMES, METHOD_COLORS, N_BOOTSTRAP,
                           OLD_METHODS, TAG_PREFIX, calculate_precision_recall_f1_topn,
                           is_any_gene_in_gold, load_new_methods, load_old_methods,
                           paired_bootstrap_aucs, paired_difference_ci)

DEPTHS = {'Total AUC': None, 'AUC@Top100': 100}
DEPTH_LABELS = list(DEPTHS)
OUT_SUBDIR = 'per_bait'


def _safe(name):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))


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


def auc_bar_all_baits(summary, out_pdf, title, method_set=None):

    baits = list(dict.fromkeys(summary['Bait']))
    methods = [m for m in (method_set or OLD_METHODS)
               if m in set(summary['Method'])]
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, len(baits), figsize=(4.2 * len(baits), 5.6),
                             sharey=True, squeeze=False)
    for ax, bait in zip(axes[0], baits):
        s = summary[summary['Bait'] == bait].set_index('Method')
        for i, depth in enumerate(DEPTH_LABELS):
            for j, m in enumerate(methods):
                if m not in s.index:
                    continue
                x = i - 0.4 + width * (j + 0.5)
                ax.bar(x, s.loc[m, depth], width, color=METHOD_COLORS[m],
                       label=m if i == 0 else None)
                ax.errorbar(x, s.loc[m, depth], yerr=s.loc[m, f'{depth} Std'],
                            fmt='none', c='black', capsize=2.5, elinewidth=1)
        ax.set_xticks(range(len(DEPTH_LABELS)))
        ax.set_xticklabels(DEPTH_LABELS, fontsize=13)
        ax.set_ylim(0, 1)
        ax.set_title(str(bait), fontsize=15)
        ax.tick_params(axis='y', labelsize=13)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel('AUC Value', fontsize=16)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Method', ncol=min(len(methods), 9),
               loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=12,
               title_fontsize=12, frameon=False)
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=[0, 0.1, 1, 0.94])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def run_dataset(cfg):
    outdir = os.path.join(cfg['outdir'], OUT_SUBDIR)
    os.makedirs(outdir, exist_ok=True)
    print(f'\n{"=" * 72}\n{cfg["name"]}\n{"=" * 72}')

    # Loaded once for the whole dataset -- identifier mapping, SFINX and
    # PP-NSAF padding and the composite min-max are dataset-wide, exactly as in
    # the pooled script, and each bait is sliced out of the result.
    features_df, frames = load_new_methods(cfg, None)
    frames.update(load_old_methods(cfg, features_df))
    frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')

    baits = list(features_df['Bait'].unique())
    identified = set(features_df['Prey'].astype(str))
    present = [m for m in ALL_METHODS if m in frames]
    print(f'  methods present: {", ".join(present)}')
    print(f'  baits: {", ".join(map(str, baits))}')

    for gt_source, gt_file in cfg['gts']:
        gt = pd.read_csv(gt_file)
        gt = gt[gt['Prey'].isin(identified)]
        gold = {b: set(gt[gt['Bait'] == b]['Prey']) for b in baits}

        summary_rows, diff_rows = [], []
        for bait in baits:
            tag = f'{TAG_PREFIX}_{gt_source}_{_safe(bait)}'

            specs, curves = [], {}
            for m in present:
                df, prey_col, score_col = frames[m]
                d = df[df['Bait'].astype(str) == str(bait)].copy()
                d['true_label'] = [is_any_gene_in_gold(p, gold.get(bait, set()))
                                   for p in d[prey_col]]
                specs.append((m, d, prey_col, score_col))
                r, p, f1, _ = calculate_precision_recall_f1_topn(
                    d, score_col, 'true_label')
                curves[m] = (r, p, f1)

            whisper_df = specs[[x[0] for x in specs].index('WHISPER')][1]
            feature_names = [f for f in FEATURE_NAMES
                             if f in whisper_df.columns]
            for feat in feature_names:
                specs.append((feat, whisper_df, 'Prey', feat))

            n_pos = int(whisper_df['true_label'].sum())
            print(f'  [{gt_source}] {bait}: {n_pos}/{len(whisper_df)} pairs '
                  f'annotated ({100 * n_pos / max(len(whisper_df), 1):.1f}%)')
            if n_pos == 0:
                print(f'    no annotated pair for {bait}; skipped')
                continue

            boot = paired_bootstrap_aucs(specs, DEPTHS,
                                         n_bootstrap=N_BOOTSTRAP, seed=0)

            def stats(name, label='Total AUC'):
                a = boot[name][label]
                return float(a.mean()), float(a.std())

            def auc_frame(methods):
                rows = []
                for m in methods:
                    if m not in boot:
                        continue
                    tot, tot_sd = stats(m, 'Total AUC')
                    top, top_sd = stats(m, 'AUC@Top100')
                    rows.append({'Method': m, 'Total AUC': tot,
                                 'Total AUC Std': tot_sd, 'AUC@Top100': top,
                                 'AUC@Top100 Std': top_sd})
                return pd.DataFrame(rows)

            def entries(methods):
                out = []
                for m in methods:
                    if m in curves and m in boot:
                        r, p, _f1 = curves[m]
                        mu, sd = stats(m)
                        out.append((m, r, p, mu, sd))
                return out

            # --- paired differences, pairs as the resampling unit -----------
            rows = []
            for lab in DEPTHS:
                for other in [m for m in present if m != 'WHISPER']:
                    row = paired_difference_ci(boot, 'WHISPER', other, lab)
                    row.update({'Bait': bait, 'comparison': f'WHISPER - {other}',
                                'reference': other, 'depth': lab,
                                'resampling_unit': 'pair',
                                'n_bootstrap': N_BOOTSTRAP,
                                'ground_truth': gt_source})
                    rows.append(row)
            pd.DataFrame(rows)[
                ['Bait', 'comparison', 'reference', 'depth', 'resampling_unit',
                 'mean_difference', 'ci_low', 'ci_high',
                 'frac_replicates_positive', 'n_bootstrap', 'ground_truth']
            ].to_csv(f'{outdir}/auc_paired_differences_{tag}.csv', index=False)
            diff_rows += rows

            # --- curve source data -----------------------------------------
            parts = []
            for m, (r, p, f1) in curves.items():
                mu, sd = stats(m)
                parts.append(pd.DataFrame({'Bait': bait, 'Method': m,
                                           'Recall': r, 'Precision': p,
                                           'F1': f1, 'AUC': mu, 'AUC_std': sd,
                                           'ground_truth': gt_source}))
            pd.concat(parts, ignore_index=True).to_csv(
                f'{outdir}/precision_recall_source_data_{tag}.csv', index=False)

            # --- every method side by side ---------------------------------
            auc_df_all = auc_frame(present)
            pr_plot(entries(present),
                    f'{outdir}/overall_precision_vs_recall_{tag}_all_methods.pdf',
                    f'{bait}: Precision vs. Recall')
            auc_bar(auc_df_all,
                    f'{outdir}/auc_barplot_{tag}_all_methods.pdf',
                    4, (0.5, -0.32), title=f'{cfg["name"]} -- {bait}')
            out_all = auc_df_all.copy()
            out_all['Bait'] = bait
            out_all['ground_truth'] = gt_source
            out_all.to_csv(f'{outdir}/auc_source_data_{tag}_all_methods.csv',
                           index=False)
            summary_rows.append(out_all)

            # --- with the individual feature baselines ----------------------
            feature_entries, feature_rows = [], []
            for feat in feature_names:
                fdf = whisper_df.dropna(subset=[feat, 'true_label'])
                if fdf.empty or feat not in boot:
                    continue
                r, p, _f1, _ = calculate_precision_recall_f1_topn(
                    fdf, feat, 'true_label')
                mu, sd = stats(feat)
                top, top_sd = stats(feat, 'AUC@Top100')
                feature_entries.append((feat, r, p, mu, sd))
                feature_rows.append({'Method': feat, 'Total AUC': mu,
                                     'Total AUC Std': sd, 'AUC@Top100': top,
                                     'AUC@Top100 Std': top_sd})

            pr_plot(entries(present),
                    f'{outdir}/overall_precision_vs_recall_{tag}_with_features.pdf',
                    f'{bait}: Including Individual Features',
                    features=feature_entries)
            auc_df_features = pd.concat(
                [auc_df_all, pd.DataFrame(feature_rows)], ignore_index=True)
            auc_bar(auc_df_features,
                    f'{outdir}/auc_barplot_{tag}_with_features.pdf',
                    4, (0.5, -0.42), title=f'{cfg["name"]} -- {bait}')
            out_feat = auc_df_features.copy()
            out_feat['Bait'] = bait
            out_feat['ground_truth'] = gt_source
            out_feat.to_csv(f'{outdir}/auc_source_data_{tag}_with_features.csv',
                            index=False)

            order = auc_df_all.sort_values('AUC@Top100', ascending=False)
            print('    Top100 ranking: ' + ', '.join(
                f'{m} {v:.3f}' for m, v in zip(order['Method'],
                                               order['AUC@Top100'])))

        # --- one table across the baits, as the per-bait figures carry -------
        if summary_rows:
            summary = pd.concat(summary_rows, ignore_index=True)
            summary.to_csv(
                f'{outdir}/auc_source_data_{TAG_PREFIX}_{gt_source}'
                f'_all_methods_per_bait.csv', index=False)
            draw_all_baits(summary, outdir, cfg['name'], gt_source)
            pd.DataFrame(diff_rows).to_csv(
                f'{outdir}/auc_paired_differences_{TAG_PREFIX}_{gt_source}'
                f'_per_bait.csv', index=False)
    print(f'  wrote {outdir}')


def draw_all_baits(summary, outdir, name, gt_source):
    for suffix, method_set in (('main_methods', OLD_METHODS),
                               ('all_methods', ALL_METHODS)):
        auc_bar_all_baits(
            summary,
            f'{outdir}/auc_barplot_{TAG_PREFIX}_{gt_source}'
            f'_{suffix}_all_baits.pdf',
            f'{name} ({gt_source})', method_set=method_set)
