# PU Learning for protein-protein interaction

This repository provides a Snakemake pipeline for reproducing results in manuscript X.

---

## Installation

Clone the repository and install the pipeline with:

```bash
git clone https://github.com/camlab-bioml/puppi_reproducibility.git
cd puppi_reproducibility
pip install -e .
```

---

## Run the pipeline with Snakemake (for example dataset 1):

```bash
snakemake --configfile config/dataset1.yaml --cores 4
```

---

## Pipeline Overview

1. **Feature Engineering**  
   Computes statistical and fold-change features for each bait–prey pair.

2. **PU Learning + FDR**  
   Trains a bagging classifier using top-N positives and bottom-K negatives; scores all pairs and estimates FDR using decoys.

3. **Ground Truth Generation**  
   Generates GO:CC or BioGRID-based ground truth for evaluation.

4. **Evaluation + Plotting**  
   Precision–recall curves, AUC comparison, and overlap with known interactions.

---

## Output Files

All major outputs are saved under `results/`:

- `features.csv` – Engineered feature matrix  
- `final_predictions.csv` – PU learning scores and FDR estimates  
- `go_cc_interactions_*.csv` – GO:CC gold standard sets  
- `biogrid_interactions_*.csv` – BioGRID-derived interaction sets  
- `overall_precision_vs_recall_combined_*.pdf` – Precision–recall curves  
- `auc_barplot_combined_*.pdf` – Barplots of AUC metrics  
- `overlap_vs_interactions_*.pdf` – Known interactor recovery analysis  
