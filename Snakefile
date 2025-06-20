configfile: "config/dataset1.yaml"

rule all:
    input:
        config["fdr_output"],
        config["gocc_output"],
        config["biogrid_output"],
        "results/Benchmarking BioID DIA/overall_precision_vs_recall_combined_bg_large_with_features.pdf",
        "results/Benchmarking BioID DIA/recovery_overlap_combined_bg_large.pdf"

rule feature_engineering:
    input:
        config["input_file"]
    output:
        config["feature_output"]
    shell:
        """
        python script.py --step feature_engineering \
            --input_file {input} --feature_output {output} \
            --control_keywords {" ".join(config["control_keywords"])}
        """

rule train_and_fdr:
    input:
        config["feature_output"]
    output:
        config["fdr_output"]
    shell:
        """
        python script.py --step train_and_fdr \
            --feature_output {input} --fdr_output {output} \
            --initial_positives {config[initial_positives]} \
            --initial_negatives {config[initial_negatives]} \
            --control_keywords {" ".join(config["control_keywords"])}
        """

rule generate_gocc:
    output:
        config["gocc_output"]
    shell:
        """
        python script.py --step generate_gocc \
            --gaf_file {config[gaf_file]} --obo_file {config[obo_file]} \
            --gocc_output {output} --baits {" ".join(config["baits"])}
        """

rule generate_biogrid:
    output:
        config["biogrid_output"]
    shell:
        """
        python script.py --step generate_biogrid \
            --biogrid_output {output} --baits {" ".join(config["baits"])}
        """

rule plot_pr:
    input:
        config["feature_output"],
        config["biogrid_output"]
    output:
        "results/Benchmarking BioID DIA/overall_precision_vs_recall_combined_bg_large_with_features.pdf"
    shell:
        """
        python script.py --step plot_pr \
            --feature_output {input[0]} \
            --plot_gt_file {input[1]} \
            --plot_gt_source bg_large
        """

rule plot_recovery:
    input:
        config["feature_output"],
        config["biogrid_output"]
    output:
        "results/Benchmarking BioID DIA/recovery_overlap_combined_bg_large.pdf"
    shell:
        """
        python script.py --step plot_recovery \
            --feature_output {input[0]} \
            --plot_gt_file {input[1]} \
            --plot_gt_source bg_large
        """
