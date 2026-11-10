"""Quality of the weak labels against the references (Supplementary Fig. 2)."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import load_uniprot_to_gene, make_to_gene
from src.train_and_fdr import strong_baits

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

N_NEGATIVES = 200
FILE_TAG = {'go_large': 'go', 'bg_large': 'bg'}


def weak_label_preys(df, initial_positives, initial_negatives=N_NEGATIVES):
    """{bait: (positive preys, negative preys)} for the strong baits."""
    df = df.sort_values(['Bait', 'Prey']).reset_index(drop=True)
    eligible = df[~df['is_inert']]
    labels = {}
    for bait in strong_baits(df, eligible):
        bait_df = eligible[eligible['Bait'] == bait]
        ranked = bait_df.sort_values('composite_score', ascending=False)
        top = ranked[ranked['single_rep_flag'] != 1].index[:initial_positives]
        bottom = bait_df.drop(index=top, errors='ignore')['composite_score'] \
            .nsmallest(initial_negatives).index
        labels[bait] = (df.loc[top, 'Prey'].astype(str).tolist(),
                        df.loc[bottom, 'Prey'].astype(str).tolist())
    return labels


def run(scores_csv, references, outdir, initial_positives, map_identifiers=False,
        gaf_file='src/goa_human.gaf'):
    whisper = pd.read_csv(scores_csv)
    whisper['Bait'] = whisper['Bait'].astype(str)
    whisper['Prey'] = whisper['Prey'].astype(str)
    to_gene = make_to_gene(load_uniprot_to_gene(gaf_file)) if map_identifiers else (lambda p: p)
    labels = weak_label_preys(whisper, initial_positives)

    for gt_source, gt_file in references.items():
        tag = FILE_TAG[gt_source]
        gt = pd.read_csv(gt_file).astype({'Bait': str, 'Prey': str})
        truth_by_bait = {b: set(sub['Prey']) for b, sub in gt.groupby('Bait')}
        rows = []
        for bait, (pos, neg) in labels.items():
            truth = truth_by_bait.get(bait, set())
            pos = [to_gene(p) for p in pos]
            neg = [to_gene(p) for p in neg]
            rows.append({'Bait': bait, 'n_positives': len(pos), 'n_negatives': len(neg),
                         'Positive_precision': sum(p in truth for p in pos) / len(pos) if pos else np.nan,
                         'Negative_contamination': sum(p in truth for p in neg) / len(neg) if neg else np.nan})
        df = pd.DataFrame(rows)
        means = [df['Positive_precision'].mean(), df['Negative_contamination'].mean()]
        sds = [df['Positive_precision'].std(), df['Negative_contamination'].std()]

        plt.figure(figsize=(5, 5))
        plt.bar(['Positives', 'Negatives'], means, yerr=sds, capsize=5,
                color=['#4C72B0', '#DD8452'])
        plt.ylim(0, 1)
        plt.ylabel('Proportion')
        plt.title('Label assignment quality (mean ± SD across baits)')
        for i, v in enumerate(means):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{outdir}/label_assignment_quality_{tag}_large.pdf', dpi=300)
        plt.close()
        df.to_csv(f'{outdir}/label_assignment_per_bait_{tag}_large.csv', index=False)
