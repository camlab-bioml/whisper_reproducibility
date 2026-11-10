# WHISPER: Weak Heuristic Inference for Supervisory Protein intERaction mapping (reproducibility)

This repository reproduces the results of the manuscript _Predicting protein interaction and proximity evidence with weakly supervised learning (Kasmaeifar et al., 2026)_ with a Snakemake workflow. It holds every input the analyses use and the result files behind the paper's figures and tables.

---

## Installation

```bash
git clone https://github.com/camlab-bioml/whisper_reproducibility.git
cd whisper_reproducibility
git lfs pull                      # the GO annotation file and the fragment-level matrix
pip install -e .
```

The deposited results were produced with Python 3.11, scikit-learn 1.5.2, pandas 2.1.4, numpy 1.26.4 and scipy 1.16.0. WHISPER's scores and FDRs are reproduced exactly with these versions; another scikit-learn release can change the random forests.

---

## Running

Each dataset is one run, and the config file is required:

```bash
snakemake --configfile config/dataset1.yaml --cores 4
```

The analyses that combine datasets run afterwards, from the results of the dataset runs:

```bash
snakemake -s cross_dataset.smk --configfile config/cross_dataset.yaml --cores 4
```

| Config | Dataset | Figures and tables |
|---|---|---|
| `dataset1.yaml` | Dataset 1, BioID, DIA, protein level | Fig. 2a, 3-6; Sup. Figs 2-13, 15-21; Sup. Tables 1, 11-15 |
| `dataset1_dda_int.yaml` | Dataset 1, DDA, intensity | Fig. 2b; Sup. Fig. 1; Sup. Table 2 |
| `dataset1_dda_spc.yaml` | Dataset 1, DDA, spectral counts | Fig. 2c; Sup. Fig. 1; Sup. Table 3 |
| `dataset1_peptide.yaml` | Dataset 1, DIA, peptide level | Fig. 2d; Sup. Fig. 1a; Sup. Table 4 |
| `dataset1_fragment.yaml` | Dataset 1, DIA, fragment level | Fig. 2e; Sup. Fig. 1a; Sup. Table 5 |
| `dataset2.yaml` | Dataset 2, BioID, DDA | Fig. 3-5; Sup. Figs 2-13, 15-20; Sup. Table 6 |
| `dataset3.yaml` | Dataset 3, BioID, DIA (proteasome) | as dataset 2; Sup. Table 7 |
| `dataset4.yaml` | Dataset 4, APEX, DDA (stress granules) | as dataset 2; Sup. Table 8 |
| `dataset5.yaml` | Dataset 5, AP-MS, DIA | as dataset 2; Sup. Table 9 |
| `mitoid.yaml` | MitoID, BioID of 17 GTPases | Sup. Fig. 14; Sup. Table 10 |
| `cross_dataset.yaml` | datasets combined | Sup. Figs 1, 12-13 (summaries), 15, 16, 17 |

The `steps` entry of each config lists the analyses run for that dataset. Single steps can be run directly (`python script.py --help` lists them):

```bash
python script.py --config config/dataset1.yaml --step benchmark
```

---

## Workflow

| Step | Code | Figure / table |
|---|---|---|
| Feature engineering | `src/feature_engineering.py` | all |
| Training and FDR | `src/train_and_fdr.py` | all; Sup. Fig. 20 (feature importance) |
| AUC against every method | `src/benchmark.py` | Fig. 2-3, Sup. Figs 9-10 |
| Overlap with the references | `src/overlap.py` | Fig. 4, Sup. Fig. 11 |
| Calls in FDR bins | `src/fdr_bins.py` | Fig. 5, Sup. Fig. 14c |
| Peptide and fragment level | `src/level_benchmark.py` | Fig. 2d-e |
| GO enrichment, HPA | `src/go_enrichment.py`, `src/hpa_annotation.py` | Fig. 6, Sup. Fig. 21, Sup. Tables 11-15 |
| Variants and int/spc models | `src/variants_similarity.py`, `src/int_spc.py` | Sup. Fig. 1 |
| Weak-label quality | `src/label_quality.py` | Sup. Fig. 2 |
| Ranking depths | `src/auc_range.py` | Sup. Figs 3-4 |
| Per bait | `src/per_bait.py` | Sup. Fig. 5 |
| Pairs both SAINT tools scored | `src/saint_intersection.py` | Sup. Fig. 6 |
| GO aspects | `src/go_aspects.py` | Sup. Fig. 7 |
| GO:CC construction | `src/gocc_sensitivity.py` | Sup. Fig. 8 |
| Bounds on the FDP | `src/fdr_bounds.py`, `src/fdr_bounds_summary.py` | Sup. Figs 12-13 |
| Out-of-fold scoring | `src/cross_validation.py` | Sup. Fig. 15 |
| Hyperparameter tuning | `src/tuning.py` | Sup. Fig. 16 |
| Transfer across datasets | `src/transfer.py` | Sup. Fig. 17 |
| Weak-label counts | `src/sensitivity.py` | Sup. Figs 18-19 |
| Supplementary tables | `src/supplementary_tables.py` | Sup. Tables 1-15 |

1. **Feature engineering**: eight features per bait-prey pair from the bait and control intensities (log fold change, signal-to-noise ratio, mean and median difference, replicate fold-change SD, bait CV, bait-to-control SD ratio, a positive fold-change flag) and the heuristic score, the mean of the four enrichment features after z-scoring within the bait. The bait's own protein and the labelling enzyme are scored but excluded from everything that shapes the result.
2. **Scoring and FDR**: baits are clustered on the spread of their top heuristic scores; each bait in the larger cluster contributes its top-ranked pairs as weak positives and its bottom 200 as weak negatives. A bagged random forest scores every pair. The FDR compares real scores with decoys made by permuting each feature within each bait, 30 times, with a running minimum so it never increases with the score.
3. **Benchmarking**: AUC of the precision-recall curve (total and top 300) against GO:CC and BioGRID with 100 paired bootstrap replicates, overlap of the top-ranked pairs with the references, and true positives and precision in FDR bins, for WHISPER, the heuristic score, SAINTq, SAINTexpress, limma, CompPASS, SFINX, PP-NSAF and MiST.

---

## Inputs (`data/`)

Each dataset folder holds the intensity matrix (`intensity.tsv`, or `.csv` for the fragment level), the outputs of the other methods (`saintq.tsv`, `saintexpress.txt`, `limma.tsv`, `comppass.csv`, `sfinx.csv`, `ppnsaf.csv`, `mist.csv`), the two references (`gocc_reference.csv`, `biogrid_reference.csv`) and the GO aspect references (`go_aspects/`). Shared inputs: `go_term_members_cache.json` (GO term members used for the GO references), `hpa_subcellular_location_all.tsv` (HPA, for the FDP lower bound) and `src/goa_human.gaf` / `src/go-basic.obo` (GO release 2024-09-08).

The other methods are run outside this workflow and their outputs are included as inputs: SAINTq and SAINTexpress with their published binaries, limma in R (one-sided moderated t-test, Benjamini-Hochberg adjusted), CompPASS with cRomppass, SFINX with the sfinx R package, MiST with the Krogan lab's R package, and PP-NSAF following Sardiu et al. (2008).

The references are the frozen copies used in the paper. They can be regenerated (`snakemake --configfile config/<dataset>.yaml --cores 1 generate_references`), which writes `*_regenerated*` beside the results for comparison. GO term members are read from the shipped cache, so the GO references rebuild offline; BioGRID is queried live (set `BIOGRID_ACCESS_KEY`) and follows the current release. UniProt accessions (datasets 3 and 4) are mapped to gene symbols offline with the vendored GAF.

---

## Results (`results/`)

`results/<dataset>/` holds the deposited WHISPER scores (`FDR_BaitSpecific_Hierarchical.csv`) and the source data and panels of every figure the dataset contributes to; `results/cross_dataset/` the analyses that combine datasets; `results/supplementary_tables/` the supplementary tables. Running the workflow regenerates them in place.
