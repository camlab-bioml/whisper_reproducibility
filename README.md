# PU Learning for High-Confidence Protein Interaction Scoring

This repository provides a modular Snakemake pipeline for identifying and evaluating high-confidence proximal protein–protein interactions (PPIs) from proximity proteomics data (e.g., BioID, AP-MS) using Positive-Unlabeled (PU) learning.

---

## Directory Structure

- `script.py` – Unified script for feature engineering, training, FDR estimation  
- `Snakefile` – Workflow definition for Snakemake  
- `config/` – Configuration files (YAML format)  
- `src/`  
  - `feature_engineering.py` – Computes features per bait–prey pair  
  - `train_and_fdr.py` – Trains PU model, generates scores, FDR estimation  
  - `generate_gocc_ground_truth.py` – Creates GO:CC ground truth 
  - `generate_biogrid_ground_truth.py` – Creates BioGRID ground truth
  - `plot_pr_curves.py` – Precision–recall and AUC comparison plots  
  - `plot_recovery_overlap.py` – Recovery of known interactions plots
- `results/` – All result files (CSV outputs, PR/AUC plots)  
- `data/` – Input files (e.g. intensity matrix)

---

## Installation

```bash
pip install -r requirements.txt
pip install goatools
```

---

## Example Config File

Create a config file, for example `config/dataset1.yaml`:

```yaml
input_file: data/intensity_matrix.csv
feature_output: results/features.csv
fdr_output: results/final_predictions.csv
control_keywords: [EGFP, Empty, NminiTurbo]
initial_positives: 10
initial_negatives: 200
```

---

## Run the pipeline with Snakemake:

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
- `biogrid_interactions.csv` – BioGRID-derived interaction sets  
- `overall_precision_vs_recall_combined_*.pdf` – Precision–recall curves  
- `auc_barplot_combined_*.pdf` – Barplots of AUC metrics  
- `overlap_vs_interactions_*.pdf` – Known interactor recovery analysis  
