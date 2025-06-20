import argparse
import yaml
import os
import pandas as pd
from src.feature_engineering import run_feature_engineering
from src.train_and_fdr import run_training_and_fdr
from src.generate_gocc_ground_truth import generate_gocc_ground_truth
from src.generate_biogrid_ground_truth import generate_biogrid_ground_truth
from src.plot_pr_curves import plot_pr_curve
from src.plot_recovery_overlap import recovery_overlap

def load_data(path):
    print(f"[INFO] Loading: {path}")
    return pd.read_csv(path)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved: {path}")

def ensure_directories_exist(paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run PU Learning Pipeline")
    parser.add_argument('--step', required=True, choices=[
        "feature_engineering", "train_and_fdr", "generate_gocc", "generate_biogrid",
        "plot_pr", "plot_recovery", "full"
    ])
    parser.add_argument('--config', required=True, help='Path to configuration YAML')

    args = parser.parse_args()

    # ✅ Load YAML config
    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    CONFIG = load_config(args.config)
    ensure_directories_exist([
        CONFIG["feature_output"],
        CONFIG["fdr_output"],
        CONFIG["gocc_output"],
        CONFIG["biogrid_output"]
    ])

    if args.step == "feature_engineering":
        df = load_data(CONFIG["input_file"])
        features_df = run_feature_engineering(df, controls=CONFIG["controls"])
        save_csv(features_df, CONFIG["feature_output"])

    elif args.step == "train_and_fdr":
        features_df = load_data(CONFIG["feature_output"])
        final_df = run_training_and_fdr(features_df, CONFIG["initial_positives"], CONFIG["initial_negatives"])
        save_csv(final_df, CONFIG["fdr_output"])

    elif args.step == "generate_gocc":
        generate_gocc_ground_truth(
            gaf_file=CONFIG["gaf_file"],
            obo_file=CONFIG["obo_file"],
            baits=CONFIG["baits"],
            output_file=CONFIG["gocc_output"]
        )

    elif args.step == "generate_biogrid":
        generate_biogrid_ground_truth(
            baits=CONFIG["baits"],
            output_file=CONFIG["biogrid_output"]
        )

    elif args.step == "plot_pr":
        features_df = load_data(CONFIG["feature_output"])
        saintq_df = load_data(CONFIG["saintq_path"])
        saintex_df = load_data(CONFIG["saintexpress_path"])
        gt_df = load_data(CONFIG["plot_gt_file"])
        plot_pr_curve(features_df, saintq_df, saintex_df, gt_df, CONFIG["plot_gt_source"])

    elif args.step == "plot_recovery":
        features_df = load_data(CONFIG["feature_output"])
        saintq_df = load_data(CONFIG["saintq_path"])
        saintex_df = load_data(CONFIG["saintexpress_path"])
        gt_df = load_data(CONFIG["plot_gt_file"])
        recovery_overlap(features_df, saintq_df, saintex_df, gt_df, CONFIG["plot_gt_source"])

    elif args.step == "full":
        df = load_data(CONFIG["input_file"])
        features_df = run_feature_engineering(df, controls=CONFIG["controls"])
        save_csv(features_df, CONFIG["feature_output"])
        final_df = run_training_and_fdr(features_df, CONFIG["initial_positives"], CONFIG["initial_negatives"])
        save_csv(final_df, CONFIG["fdr_output"])

if __name__ == "__main__":
    main()
