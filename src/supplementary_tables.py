"""Supplementary Tables 1-15."""

import os

import pandas as pd

# Column descriptions shared across datasets, keyed by sheet name.
COLUMNS = {
    'WHISPER': [
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein (gene name).'),
        ('log_fold_change', 'Log2 ratio of mean bait to mean control intensity, divided by the number of zero-valued bait replicates (when there is at least one).'),
        ('snr', 'Mean bait intensity divided by the standard deviation of control intensities, divided by the number of zero-valued bait replicates (when there is at least one).'),
        ('mean_diff', 'Mean bait intensity minus mean control intensity.'),
        ('median_diff', 'Median bait intensity minus median control intensity.'),
        ('replicate_fold_change_sd', 'Standard deviation of bait replicate intensities divided by the mean control intensity.'),
        ('bait_cv', 'Coefficient of variation of bait replicate intensities.'),
        ('bait_control_sd_ratio', 'Standard deviation of bait intensities divided by the standard deviation of control intensities.'),
        ('zero_or_neg_fc', '1 if log_fold_change is positive, 0 otherwise.'),
        ('nonzero_reps', 'Number of bait replicates with non-zero intensity.'),
        ('reps_above_ctrl_med', 'Number of bait replicates with intensity above the median control intensity.'),
        ('single_rep_flag', '1 if the prey has non-zero intensity in only one bait replicate; such preys are not used as initial positives.'),
        ('is_inert', "True for the bait's own protein and the labelling enzyme, which are excluded from labelling, decoys and FDR estimation."),
        ('composite_score', 'Heuristic score: mean of the per-bait z-scored log_fold_change, snr, mean_diff and median_diff.'),
        ('global_cv', 'Coefficient of variation of the prey intensity across all bait and control samples.'),
        ('predicted_probability', 'WHISPER predicted probability of interaction.'),
        ('FDR', 'WHISPER FDR estimated from the decoy null.'),
        ('global_cv_flag', "'likely background' if global_cv is at or below the 25th percentile across all pairs."),
    ],
    'SAINTq': [
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein (gene name).'),
        ('#Rep', 'Number of bait replicates.'),
        ('#Pep', 'Number of peptides used for the protein.'),
        ('#Frag', 'Number of fragments used for each peptide, separated by |.'),
        ('AvgP', 'Average probability of interaction across replicates.'),
        ('BFDR', 'Bayesian FDR.'),
    ],
    'SAINTexpress': [
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein.'),
        ('PreyGene', 'Prey gene name.'),
        ('Intensity', "Bait replicate intensities as reported by SAINTexpress ('.' = not detected)."),
        ('IntensitySum', 'Sum of bait replicate intensities.'),
        ('AvgIntensity', 'Mean bait replicate intensity.'),
        ('NumReplicates', 'Number of bait replicates.'),
        ('ctrlIntensity', "Control intensities ('.' = not detected)."),
        ('AvgP', 'Average probability of interaction across replicates.'),
        ('MaxP', 'Maximum probability of interaction across replicates.'),
        ('TopoAvgP', 'AvgP after topology-based boosting.'),
        ('TopoMaxP', 'MaxP after topology-based boosting.'),
        ('SaintScore', 'SAINT score.'),
        ('OddsScore', 'Log odds score.'),
        ('Spec', "Bait replicate spectral counts ('.' = not detected)."),
        ('SpecSum', 'Sum of bait replicate spectral counts.'),
        ('AvgSpec', 'Mean bait replicate spectral count.'),
        ('ctrlCounts', 'Control spectral counts.'),
        ('logOddsScore', 'Log odds score.'),
        ('FoldChange', 'Fold change of bait over control intensity.'),
        ('BFDR', 'Bayesian FDR.'),
        ('boosted_by', 'Preys that boosted the score through topology.'),
    ],
    'limma': [
        ('logFC', 'Log2 fold change of bait over pooled controls (log2(intensity + 1)).'),
        ('AveExpr', 'Average log2 intensity across all samples.'),
        ('t', 'Moderated t-statistic (used for ranking).'),
        ('P.Value', 'Two-sided p-value.'),
        ('adj.P.Val', 'Benjamini-Hochberg adjusted two-sided p-value within the bait.'),
        ('B', 'Log odds of differential abundance.'),
        ('P.Value.one_sided', 'One-sided p-value for enrichment (P/2 if t > 0, otherwise 1 - P/2).'),
        ('adj.P.Val.one_sided', 'Benjamini-Hochberg adjusted one-sided p-value within the bait (used for calls).'),
        ('Protein', 'Prey protein (gene name).'),
        ('Bait', 'Bait gene name.'),
    ],
    'CompPASS': [
        ('Experiment.ID', 'Experiment identifier.'),
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein (gene name).'),
        ('AvePSM', 'Mean bait replicate abundance (intensity or spectral count, as in the input data).'),
        ('Z', 'Z-score of the prey abundance across baits.'),
        ('WD', 'WD score (used for ranking).'),
        ('Entropy', 'Entropy of the prey abundance across replicates.'),
    ],
    'SFINX': [
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein (gene name).'),
        ('Scores', 'SFINX score (lower is more confident).'),
        ('pValues', 'SFINX p-value.'),
        ('sfinx_score', '-log10(Scores), used for ranking. Only pairs retained by SFINX are listed.'),
    ],
    'MiST': [
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein (gene name).'),
        ('mist_score', 'MiST score (used for ranking).'),
        ('Reproducibility', 'Reproducibility component.'),
        ('Abundance', 'Abundance component.'),
        ('Specificity18', 'Standardized specificity component (MiST v1.5 default).'),
    ],
    'PP-NSAF': [
        ('Bait', 'Bait gene name.'),
        ('Prey', 'Prey protein (gene name).'),
        ('NSAF', 'Normalized abundance factor, without protein-length normalization.'),
        ('ppnsaf_score', 'PP-NSAF posterior probability (used for ranking).'),
    ],
}


TITLES = {
    1: 'Supplementary Table 1. Protein-level scoring output of WHISPER and the benchmarking methods for BioID dataset 1 acquired in DIA mode.',
    2: 'Supplementary Table 2. Protein-level scoring output of WHISPER and the benchmarking methods for BioID dataset 1 acquired in DDA mode and quantified by intensity (IonQuant).',
    3: 'Supplementary Table 3. Protein-level scoring output of WHISPER and the benchmarking methods for BioID dataset 1 acquired in DDA mode and quantified by spectral counts. SAINTq and limma were not applied to spectral counts.',
    4: 'Supplementary Table 4. Peptide-level scoring of BioID dataset 1 acquired in DIA mode, aggregated to the protein level, for WHISPER and SAINTq.',
    5: 'Supplementary Table 5. Fragment-level scoring of BioID dataset 1 acquired in DIA mode, aggregated to the protein level, for WHISPER and SAINTq.',
    6: 'Supplementary Table 6. Protein-level scoring output of WHISPER and the benchmarking methods for BioID dataset 2 (DDA, intensity).',
    7: 'Supplementary Table 7. Protein-level scoring output of WHISPER and the benchmarking methods for BioID dataset 3 (proteasome, DIA).',
    8: 'Supplementary Table 8. Protein-level scoring output of WHISPER and the benchmarking methods for the APEX dataset (stress granules, DDA).',
    9: 'Supplementary Table 9. Protein-level scoring output of WHISPER and the benchmarking methods for the AP-MS dataset (DIA).',
    10: 'Supplementary Table 10. Protein-level scoring output of WHISPER and the benchmarking methods for the MitoID dataset (17 GTP-locked GTPases, DDA).',
}
SHEET_ORDER = [('SAINTq', 'saintq'), ('SAINTexpress', 'saintexpress'), ('limma', 'limma'),
               ('CompPASS', 'comppass'), ('SFINX', 'sfinx'), ('MiST', 'mist'),
               ('PP-NSAF', 'ppnsaf')]


def _level_columns(unit):
    return {'WHISPER': [
        ('Protein', 'Prey protein (gene name).'),
        ('predicted_probability', f"Maximum WHISPER predicted probability across the protein's {unit}s."),
        ('FDR', f'WHISPER protein-level FDR, estimated from the maximum probability per protein for real and decoy {unit}s.'),
        ('global_cv_flag', f"'likely background' if at least half of the protein's {unit}s are flagged as likely background (global CV at or below the 25th percentile)."),
    ]}


def table_spec(num, cfg):
    """The sheets of table `num` from its dataset config."""
    level = cfg.get('level', 'protein')
    if level == 'protein':
        whisper = os.path.join(cfg['results_dir'], 'FDR_BaitSpecific_Hierarchical.csv')
        sheets = [('WHISPER', whisper)] + [(name, cfg['methods'][key]) for name, key in SHEET_ORDER
                                           if key in cfg.get('methods', {})]
        return dict(title=TITLES[num], sheets=sheets)
    whisper = os.path.join(cfg['results_dir'], cfg['protein_scores_file'])
    return dict(title=TITLES[num], sheets=[('WHISPER', whisper), ('SAINTq', cfg['methods']['saintq'])],
                columns=_level_columns(level),
                rename={'WHISPER': {'background_flag_protein': 'global_cv_flag'}})


def read_output(path):
    sep = ',' if path.endswith('.csv') else '\t'
    # keep_default_na=False leaves empty strings and gene names such as 'NA'
    # as they are in the source file.
    return pd.read_csv(path, sep=sep, keep_default_na=False, na_values=[''])


def description(table, frames):
    rows = [(table['title'], ''), ('', '')]
    for name, df in frames:
        rows.append((f'Sheet: {name}', f'{len(df)} rows'))
        known = dict(COLUMNS.get(name, []))
        known.update(dict(table.get('columns', {}).get(name, [])))
        for col in df.columns:
            rows.append((col, known.get(col, '')))
        rows.append(('', ''))
    return pd.DataFrame(rows, columns=['Item', 'Description'])


def build(num, cfg, out_dir):
    table = table_spec(num, cfg)
    frames = [(name, read_output(path).rename(columns=table.get('rename', {}).get(name, {})))
              for name, path in table['sheets']]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'Supplementary Table {num}.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        description(table, frames).to_excel(xw, sheet_name='Description', index=False)
        for name, df in frames:
            df.to_excel(xw, sheet_name=name, index=False)


GO_TABLES = {11: 'ACTB', 12: 'CTNNA1', 13: 'KRT8', 14: 'LMNA', 15: 'MAPRE3'}
GO_COLUMNS = [
    ('source', 'Annotation source (GO:CC).'),
    ('native', 'GO term identifier.'),
    ('name', 'GO term name.'),
    ('description', 'GO term definition.'),
    ('p_value', 'g:Profiler adjusted p-value (g:SCS correction).'),
    ('significant', 'True if the adjusted p-value is below 0.05.'),
    ('term_size', 'Number of background proteins annotated to the term.'),
    ('query_size', 'Number of query proteins recognised by g:Profiler.'),
    ('intersection_size', 'Number of query proteins annotated to the term.'),
    ('intersection', 'Query proteins annotated to the term.'),
    ('effective_domain_size', 'Number of background proteins (proteins detected in the dataset) recognised by g:Profiler.'),
    ('precision', 'intersection_size / query_size.'),
    ('recall', 'intersection_size / term_size.'),
    ('Bait', 'Bait gene name.'),
    ('Method', 'Method, or the combination of methods whose shared preys form the subset.'),
]


def whisper_name(label):
    return '+'.join('WHISPER' if m == 'puppi' else m for m in str(label).split('+'))


def sheet_title(label):
    return label if len(label) <= 31 else label.replace('WHISPER', 'WHISP')[:31]


def build_go(num, go_dir, out_dir):
    bait = GO_TABLES[num]
    overall = pd.read_excel(f'{go_dir}/overall_enrichment/{bait}_gprofiler.xlsx', sheet_name=None)
    upset = pd.read_excel(f'{go_dir}/upset_enrichment/{bait}_upset_gprofiler.xlsx', sheet_name=None)
    order = ['WHISPER', 'limma', 'SAINTq', 'SAINTexpress']
    sheets = []
    for label, df in sorted(overall.items(), key=lambda kv: order.index(whisper_name(kv[0]))):
        sheets.append((f'{whisper_name(label)} - all preys', df))
    for label, df in sorted(upset.items(), key=lambda kv: (-len(kv[0].split('+')), whisper_name(kv[0]))):
        sheets.append((whisper_name(label), df))
    for _, df in sheets:
        if 'Method' in df.columns:
            df['Method'] = df['Method'].map(whisper_name)
    rows = [(f'Supplementary Table {num}. GO:CC enrichment of high-confidence {bait} preys (1% FDR) in BioID dataset 1, tested with g:Profiler against the proteins detected in the dataset.', ''),
            ('', ''),
            ("Sheets named '<method> - all preys'", "Enrichment of all preys called by that method at 1% FDR (limma: one-sided adjusted p-value). Methods with too few calls for enrichment have no sheet."),
            ('Other sheets', "Enrichment of the preys called exclusively by the listed combination of methods (one UpSet intersection), e.g. 'WHISPER' = called by WHISPER only. Empty sheets mean no significant term."),
            ('', '')]
    for name, df in sheets:
        rows.append((f'Sheet: {name}', f'{len(df)} terms'))
    rows.append(('', ''))
    rows += GO_COLUMNS
    out = os.path.join(out_dir, f'Supplementary Table {num}.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        pd.DataFrame(rows, columns=['Item', 'Description']).to_excel(xw, sheet_name='Description', index=False)
        for name, df in sheets:
            df.to_excel(xw, sheet_name=sheet_title(name), index=False)
    print(f'Wrote {out}')
