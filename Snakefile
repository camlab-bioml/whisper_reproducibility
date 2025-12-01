# Snakefile

configfile: "config.yaml"  # can be overridden via --configfile

# -------------------------
# Core data prep & training
# -------------------------

rule feature_engineering:
    input:
        data=config["input_file"]
    output:
        features=config["feature_output"]
    params:
        controls=",".join(config["controls"])
    shell:
        """
        python script.py --step feature_engineering --config {configfile}
        """

rule train_and_fdr:
    input:
        features=config["feature_output"]
    output:
        fdr=config["fdr_output"]
    shell:
        """
        python script.py --step train_and_fdr --config {configfile}
        """

# -------------------------
# Ground-truth generation
# -------------------------

rule generate_gocc:
    output:
        config["gocc_output"]
    shell:
        """
        python script.py --step generate_gocc --config {configfile}
        """

rule generate_biogrid:
    output:
        config["biogrid_output"]
    shell:
        """
        python script.py --step generate_biogrid --config {configfile}
        """

# -------------------------
# Plots & analyses
# -------------------------

# Precision–Recall curves (aggregated)
rule plot_pr:
    input:
        # Ensure features exist and external method tables + GT are available
        features=config["feature_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        gt=config["plot_gt_file"]
    shell:
        """
        python script.py --step plot_pr --config {configfile}
        """

# Recovery/overlap plots
rule plot_recovery:
    input:
        features=config["feature_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        gt=config["plot_gt_file"]
    shell:
        """
        python script.py --step plot_recovery --config {configfile}
        """

# Precision & TP by FDR bins (whisper vs SAINTq/express vs limma)
rule plot_fdr_bins:
    input:
        # Depend on trained whisper (FDR present), external methods, and GT
        fdr=config["fdr_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        limma=config["limma_path"],
        gt=config["plot_gt_file"]
    shell:
        """
        python script.py --step plot_fdr_bins --config {configfile}
        """

# FDR robustness distributions (whisper FDR vs alternative nulls)
rule plot_fdr_robustness:
    input:
        # Use trained whisper output containing probabilities & FDR
        fdr=config["fdr_output"]
    shell:
        """
        python script.py --step plot_fdr_robustness --config {configfile}
        """

# Sensitivity sweep over initial positives/negatives (training inside)
rule plot_sensitivity:
    input:
        # Needs engineered features (training happens within the step)
        features=config["feature_output"]
    shell:
        """
        python script.py --step plot_sensitivity --config {configfile}
        """

# PR AUC sweep across positives/negatives (training inside)
rule plot_posneg_pr:
    input:
        features=config["feature_output"],
        gt=config["plot_gt_file"]
    shell:
        """
        python script.py --step plot_posneg_pr --config {configfile}
        """


# GO:CC gProfiler + UpSet analysis per bait
rule plot_go_upset:
    input:
        fdr=config["fdr_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        limma=config["limma_path"]
    shell:
        """
        python script.py --step plot_go_upset --config {configfile}
        """


# HPA-based circular plots + stacked barplots
rule plot_hpa:
    input:
        fdr=config["fdr_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        limma=config["limma_path"],
        hpa=config["hpa_path"]
    shell:
        """
        python script.py --step plot_hpa --config {configfile}
        """

        
# -------------------------
# Full pipeline (data prep + training)
# -------------------------
rule full_pipeline:
    input:
        fdr=config["fdr_output"]
    shell:
        """
        python script.py --step full --config {configfile}
        """
