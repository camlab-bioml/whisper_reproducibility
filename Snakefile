# WHISPER reproducibility workflow. One dataset per run:
#
#     snakemake --configfile config/dataset1.yaml --cores 4
#
# Every rule calls script.py with the same config file; `steps` in the config
# selects which figures and tables the dataset contributes to (see README). The
# analyses that combine datasets are in cross_dataset.smk.

# No default config: a default would be merged key by key into the one given
# on the command line, and a dataset would inherit settings it does not have.
if not workflow.configfiles:
    raise ValueError("Give a dataset config: snakemake --configfile config/<dataset>.yaml")
CONFIG_FILE = workflow.configfiles[-1]
R = config["results_dir"].rstrip("/")
LEVEL = config.get("level", "protein")
REFS = list(config["references"])                       # go_large, bg_large
REF_FILES = list(config["references"].values())
METHOD_FILES = list(config.get("methods", {}).values())
CALL_FILES = [v for k, v in config.get("methods", {}).items()
              if k in ("saintq", "saintexpress", "limma")]     # methods with FDRs
SCORES = f"{R}/{config.get('scores_file', 'FDR_BaitSpecific_Hierarchical.csv')}"
FEATURES = f"{R}/features.csv"
STEPS = config.get("steps", [])
TAG = {"go_large": "go", "bg_large": "bg"}
TABLES = "results/supplementary_tables"
GO_RELEASE_2021 = "data/go_release_2021-01-01"
HPA_ALL = "data/hpa_subcellular_location_all.tsv"
RUN = f"python script.py --config {CONFIG_FILE} --step"

STEP_OUTPUTS = {
    "benchmark": expand(f"{R}/auc_source_data_all_{{ref}}_main.csv", ref=REFS),
    "overlap": expand(f"{R}/overlap_source_data_all_{{ref}}.csv", ref=REFS),
    "fdr_bins": expand(f"{R}/tp_precision_source_data_{{ref}}.csv", ref=REFS),
    "label_quality": [f"{R}/label_assignment_per_bait_{TAG[r]}_large.csv" for r in REFS],
    "auc_range": expand(f"{R}/auc_range_source_data_{{ref}}.csv", ref=REFS),
    "per_bait": expand(f"{R}/per_bait/auc_source_data_all_{{ref}}_all_methods_per_bait.csv", ref=REFS),
    "saint_intersection": expand(f"{R}/intersection/auc_source_data_all_{{ref}}_main.csv", ref=REFS),
    "go_aspects": [f"{R}/go_aspects/auc_by_aspect.csv"],
    "gocc_sensitivity": [f"{R}/gocc_sensitivity/auc_gocc_sensitivity_one_row.pdf"],
    "fdr_bounds_methods": [f"{R}/fdr_bounds_contaminants_lumen/fdr_bounds_contaminants.csv"],
    "fdr_bounds_nulls": [f"{R}/fdr_bounds_nulls/fdr_bounds_nulls.csv"],
    "cross_validation": [f"{R}/FDR_BaitSpecific_Hierarchical_CV.csv"],
    "tuning_sweep": [f"{R}/hyperparameter_tuning/hp_internal.csv"],
    "tuned_model": expand(f"{R}/bestconfig/auc_source_data_all_{{ref}}_all_methods.csv", ref=REFS),
    "posneg_auc": [f"{R}/puppi_pr_results_by_pos_neg_{TAG[r]}.csv" for r in REFS],
    "label_diagnostics": [f"{R}/pu_learning_optimization.csv"],
    "level_benchmark": expand(f"{R}/auc_source_data_{{ref}}_main_methods_{LEVEL}.csv", ref=REFS),
    "go_enrichment": [f"{R}/go_enrichment/.done"],
    "hpa": [f"{R}/hpa/{config.get('hpa_bait', '')}_HPA_category_composition_stacked_numbers.pdf"],
    "supplementary_table": [f"{TABLES}/Supplementary Table {config.get('table_number', 0)}.xlsx"],
    "go_tables": [f"{TABLES}/Supplementary Table {n}.xlsx" for n in range(11, 16)],
}
TRAIN_OUTPUTS = [SCORES, f"{R}/feature_importance_model_based.csv"]
if LEVEL != "protein":
    TRAIN_OUTPUTS.append(f"{R}/{config['protein_scores_file']}")
TABLE_SCORES = TRAIN_OUTPUTS[-1] if LEVEL != "protein" else SCORES


rule all:
    input:
        TRAIN_OUTPUTS,
        [f for step in STEPS for f in STEP_OUTPUTS[step]]


# -------------------------------------------------------------------------
# WHISPER scoring
# -------------------------------------------------------------------------

rule feature_engineering:
    input: config["input_file"]
    output: FEATURES
    shell: f"{RUN} feature_engineering"

rule train_and_fdr:            # also Sup. Fig. 20
    input: FEATURES
    output: TRAIN_OUTPUTS
    shell: f"{RUN} train_and_fdr"


# -------------------------------------------------------------------------
# Benchmarking against the other methods
# -------------------------------------------------------------------------

rule benchmark:                # Fig. 2-3, Sup. Figs 9-10
    input: SCORES, METHOD_FILES, REF_FILES
    output: STEP_OUTPUTS["benchmark"]
    shell: f"{RUN} benchmark"

rule overlap:                  # Fig. 4, Sup. Fig. 11
    input: SCORES, METHOD_FILES, REF_FILES
    output: STEP_OUTPUTS["overlap"]
    shell: f"{RUN} overlap"

rule fdr_bins:                 # Fig. 5, Sup. Fig. 14c
    input: SCORES, CALL_FILES, REF_FILES
    output: STEP_OUTPUTS["fdr_bins"]
    shell: f"{RUN} fdr_bins"

rule auc_range:                # Sup. Figs 3-4
    input: SCORES, METHOD_FILES, REF_FILES
    output: STEP_OUTPUTS["auc_range"]
    shell: f"{RUN} auc_range"

rule per_bait:                 # Sup. Fig. 5
    input: SCORES, METHOD_FILES, REF_FILES
    output: STEP_OUTPUTS["per_bait"]
    shell: f"{RUN} per_bait"

rule saint_intersection:       # Sup. Fig. 6
    input: SCORES, METHOD_FILES, REF_FILES
    output: STEP_OUTPUTS["saint_intersection"]
    shell: f"{RUN} saint_intersection"

rule go_aspects:               # Sup. Fig. 7
    input: SCORES, METHOD_FILES, config.get("go_aspects_dir", [])
    output: STEP_OUTPUTS["go_aspects"]
    shell: f"{RUN} go_aspects"

rule gocc_sensitivity:         # Sup. Fig. 8
    input: SCORES, METHOD_FILES, f"{GO_RELEASE_2021}/goa_human.gaf", f"{GO_RELEASE_2021}/go-basic.obo"
    output: STEP_OUTPUTS["gocc_sensitivity"]
    shell: f"{RUN} gocc_sensitivity"

rule level_benchmark:          # Fig. 2d-e
    input: SCORES, CALL_FILES, REF_FILES
    output: STEP_OUTPUTS["level_benchmark"]
    shell: f"{RUN} level_benchmark"


# -------------------------------------------------------------------------
# The weak labels, the model and its FDR
# -------------------------------------------------------------------------

rule label_quality:            # Sup. Fig. 2
    input: SCORES, REF_FILES
    output: STEP_OUTPUTS["label_quality"]
    shell: f"{RUN} label_quality"

rule fdr_bounds_methods:       # Sup. Fig. 12
    input: SCORES, CALL_FILES, REF_FILES, HPA_ALL
    output: STEP_OUTPUTS["fdr_bounds_methods"]
    shell: f"{RUN} fdr_bounds_methods"

rule fdr_bounds_nulls:         # Sup. Fig. 13
    input: FEATURES, REF_FILES, HPA_ALL
    output: STEP_OUTPUTS["fdr_bounds_nulls"]
    shell: f"{RUN} fdr_bounds_nulls"

rule cross_validation:         # Sup. Fig. 15 (evaluated in cross_dataset.smk)
    input: FEATURES
    output: STEP_OUTPUTS["cross_validation"]
    shell: f"{RUN} cross_validation"

rule tuning_sweep:             # Sup. Fig. 16
    input: FEATURES
    output: STEP_OUTPUTS["tuning_sweep"]
    shell: f"{RUN} tuning_sweep"

rule tuned_model:              # Sup. Fig. 16
    input: FEATURES, STEP_OUTPUTS["tuning_sweep"], METHOD_FILES, REF_FILES
    output: STEP_OUTPUTS["tuned_model"]
    shell: f"{RUN} tuned_model"

rule posneg_auc:               # Sup. Fig. 18
    input: FEATURES, REF_FILES
    output: STEP_OUTPUTS["posneg_auc"]
    shell: f"{RUN} posneg_auc"

rule label_diagnostics:        # Sup. Fig. 19
    input: FEATURES
    output: STEP_OUTPUTS["label_diagnostics"]
    shell: f"{RUN} label_diagnostics"


# -------------------------------------------------------------------------
# Case study (dataset 1)
# -------------------------------------------------------------------------

rule go_enrichment:            # Fig. 6a-c (needs internet)
    input: SCORES, CALL_FILES
    output: touch(STEP_OUTPUTS["go_enrichment"][0])
    shell: f"{RUN} go_enrichment"

rule hpa:                      # Fig. 6d-e, Sup. Fig. 21
    input: SCORES, CALL_FILES, config.get("hpa_path", [])
    output: STEP_OUTPUTS["hpa"]
    shell: f"{RUN} hpa"


# -------------------------------------------------------------------------
# Supplementary tables
# -------------------------------------------------------------------------

rule supplementary_table:      # Sup. Tables 1-10
    input: TABLE_SCORES, METHOD_FILES
    output: STEP_OUTPUTS["supplementary_table"]
    shell: f"{RUN} supplementary_table"

rule go_tables:                # Sup. Tables 11-15
    input: STEP_OUTPUTS["go_enrichment"]
    output: STEP_OUTPUTS["go_tables"]
    shell: f"{RUN} go_tables"


# -------------------------------------------------------------------------
# Inputs fetched from permanent archives
# -------------------------------------------------------------------------

rule download_go_2021:         # the older GO release of Sup. Fig. 8
    output:
        gaf=f"{GO_RELEASE_2021}/goa_human.gaf",
        obo=f"{GO_RELEASE_2021}/go-basic.obo"
    shell:
        "curl -sL https://release.geneontology.org/2021-01-01/annotations/goa_human.gaf.gz"
        " | gunzip -c > {output.gaf} && "
        "curl -sL https://release.geneontology.org/2021-01-01/ontology/go-basic.obo -o {output.obo}"


# -------------------------------------------------------------------------
# Optional: regenerate the references. The figures use the frozen copies in
# data/; these write fresh ones to the results folder for comparison.
#     snakemake --configfile config/dataset1.yaml --cores 1 generate_references
# -------------------------------------------------------------------------

rule generate_gocc:
    output: f"{R}/gocc_reference_regenerated.csv"
    shell: f"{RUN} generate_gocc"

rule generate_biogrid:
    output: f"{R}/biogrid_reference_regenerated.csv"
    shell: f"{RUN} generate_biogrid"

rule generate_go_aspects:
    output: f"{R}/go_aspects_regenerated/reference_summary.csv"
    shell: f"{RUN} generate_go_aspects"

rule generate_references:
    input:
        f"{R}/gocc_reference_regenerated.csv",
        f"{R}/biogrid_reference_regenerated.csv",
        f"{R}/go_aspects_regenerated/reference_summary.csv"
