# Snakefile

configfile: "config.yaml"  # can be overridden via --configfile

# Rule to run feature engineering
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

# Rule to train PU model and estimate FDR
rule train_and_fdr:
    input:
        features=config["feature_output"]
    output:
        fdr=config["fdr_output"]
    shell:
        """
        python script.py --step train_and_fdr --config {configfile}
        """

# Rule to generate GO:CC ground truth
rule generate_gocc:
    output:
        config["gocc_output"]
    shell:
        """
        python script.py --step generate_gocc --config {configfile}
        """

# Rule to generate BioGRID ground truth
rule generate_biogrid:
    output:
        config["biogrid_output"]
    shell:
        """
        python script.py --step generate_biogrid --config {configfile}
        """

# Rule to plot precision-recall curves
rule plot_pr:
    input:
        fdr=config["fdr_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        gt=config["plot_gt_file"]
    shell:
        """
        python script.py --step plot_pr --config {configfile}
        """

# Rule to plot overlap recovery
rule plot_recovery:
    input:
        fdr=config["fdr_output"],
        saintq=config["saintq_path"],
        saintex=config["saintexpress_path"],
        gt=config["plot_gt_file"]
    shell:
        """
        python script.py --step plot_recovery --config {configfile}
        """

# Full pipeline rule (optional if you want everything in one go)
rule full_pipeline:
    input:
        fdr=config["fdr_output"]
    shell:
        """
        python script.py --step full --config {configfile}
        """
