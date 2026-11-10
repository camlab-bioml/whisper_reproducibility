# Analyses that combine datasets. Run after every dataset config:
#
#     snakemake -s cross_dataset.smk --configfile config/cross_dataset.yaml --cores 4
#
# Each rule reads the per-dataset results the dataset runs wrote; a missing one
# means that dataset's run (or the step listed in its config) has not been done.

import yaml

if not workflow.configfiles:
    raise ValueError("Give the cross-dataset config: --configfile config/cross_dataset.yaml")
CONFIG_FILE = workflow.configfiles[-1]
R = config["results_dir"].rstrip("/")
RUN = f"python script.py --config {CONFIG_FILE} --step"
CFG = {k: yaml.safe_load(open(p)) for k, p in config["datasets"].items()}
BENCH = config["benchmark_datasets"]


def res(key, name):
    return f"{CFG[key]['results_dir'].rstrip('/')}/{name}"


rule all:
    input:
        f"{R}/fdr_bounds_summary/fdr_bounds_nulls_summary.csv",
        f"{R}/hyperparameter_comparison/cv_per_fold_summary.csv",
        f"{R}/hyperparameter_comparison/bestconfig_vs_vanilla.csv",
        f"{R}/transfer/transfer_summary.csv",
        f"{R}/variants_similarity/whisper_Variants_Similarity/average_spearman_across_baits.csv",
        f"{R}/int_plus_spc/int_spc_auc_source_data.csv"

rule fdr_bounds_summary:       # Sup. Figs 12-13, every dataset at 1% and 5%
    input:
        [res(k, "fdr_bounds_contaminants_lumen/fdr_bounds_contaminants.csv") for k in BENCH],
        [res(k, "fdr_bounds_nulls/fdr_bounds_nulls.csv") for k in BENCH]
    output: f"{R}/fdr_bounds_summary/fdr_bounds_nulls_summary.csv"
    shell: f"{RUN} fdr_bounds_summary"

rule cv_per_fold:              # Sup. Fig. 15
    input: [res(k, "FDR_BaitSpecific_Hierarchical_CV.csv") for k in BENCH]
    output: f"{R}/hyperparameter_comparison/cv_per_fold_summary.csv"
    shell: f"{RUN} cv_per_fold"

rule tuning_compare:           # Sup. Fig. 16
    input:
        [res(k, "hyperparameter_tuning/hp_internal.csv") for k in BENCH],
        [res(k, "bestconfig/auc_source_data_all_go_large_all_methods.csv") for k in BENCH]
    output: f"{R}/hyperparameter_comparison/bestconfig_vs_vanilla.csv"
    shell: f"{RUN} tuning_compare"

rule transfer:                 # Sup. Fig. 17
    input:
        res(config["transfer"]["source"], "features.csv"),
        [res(k, "FDR_BaitSpecific_Hierarchical.csv") for k in config["transfer"]["targets"]]
    output: f"{R}/transfer/transfer_summary.csv"
    shell: f"{RUN} transfer"

rule variants_similarity:      # Sup. Fig. 1a
    input:
        [res(k, CFG[k].get("protein_scores_file", "FDR_BaitSpecific_Hierarchical.csv"))
         for k, kind in config["variants"]["whisper"].values()]
    output: f"{R}/variants_similarity/whisper_Variants_Similarity/average_spearman_across_baits.csv"
    shell: f"{RUN} variants_similarity"

rule int_spc:                  # Sup. Fig. 1b
    input:
        res(config["int_spc"]["intensity"], "features.csv"),
        res(config["int_spc"]["spectral_counts"], "features.csv")
    output: f"{R}/int_plus_spc/int_spc_auc_source_data.csv"
    shell: f"{RUN} int_spc"
