configfile: "config/dataset1.yaml"

rule all:
    input:
        config["fdr_output"]

rule feature_engineering:
    input:
        config["input_file"]
    output:
        config["feature_output"]
    params:
        control_keywords=" ".join(config["control_keywords"])
    shell:
        """
        python script.py \
            --step feature_engineering \
            --input_file {input} \
            --feature_output {output} \
            --control_keywords {params.control_keywords}
        """

rule train_and_fdr:
    input:
        config["feature_output"]
    output:
        config["fdr_output"]
    shell:
        """
        python script.py \
            --step train_and_fdr \
            --input_file {input} \
            --feature_output {input} \
            --fdr_output {output} \
            --initial_positives {config[initial_positives]} \
            --initial_negatives {config[initial_negatives]} \
            --control_keywords {" ".join(config["control_keywords"])}
        """
