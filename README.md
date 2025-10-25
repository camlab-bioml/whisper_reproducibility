# WHISPER: Weak Heuristic Inference for Supervisory Protein intERaction mapping (reproducibility)

This repository provides a Snakemake pipeline for reproducing results in manuscript X.

---

## Installation

Clone the repository and install the pipeline with:

```bash
git clone https://github.com/camlab-bioml/whisper_reproducibility.git
cd whisper_reproducibility
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

2. **Scoring + FDR**  
   Trains a bagging classifier using top-N positives and bottom-K negatives; scores all pairs and estimates FDR using decoys.

3. **Ground Truth Generation**  
   Generates GO:CC or BioGRID-based ground truth for evaluation.

4. **Evaluation + Plotting**  
   All main and supplementary figures in the manuscript including: Precision–recall curves, overlap with known interactions, precision across FDR bins, sensitivity plots.

---

