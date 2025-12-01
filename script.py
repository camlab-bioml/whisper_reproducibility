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
from src.plot_fdr_bins import plot_fdr_bins
from src.plot_sensitivity import plot_sensitivity
from src.plot_posneg_pr import plot_posneg_pr
from src.plot_fdr_robustness import plot_fdr_robustness
from src.plot_go_upset import run_go_upset_analysis
from src.plot_hpa_annotations import plot_hpa_annotations


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
    parser = argparse.ArgumentParser(description="Run WHISPER Pipeline")
    parser.add_argument('--step', required=True, choices=[
        "feature_engineering", "train_and_fdr", "generate_gocc", "generate_biogrid",
        "plot_pr", "plot_recovery", "plot_fdr_bins", "plot_fdr_robustness",
        "plot_sensitivity", "plot_posneg_pr", "plot_go_upset", "plot_hpa", "full"
    ])
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    args = parser.parse_args()

    def load_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    CONFIG = load_config(args.config)
    ensure_directories_exist([
        CONFIG["feature_output"],
        CONFIG["fdr_output"],
        CONFIG["gocc_output"],
        CONFIG["biogrid_output"],
        CONFIG.get("results_dir", "results")   # <-- ensure results_dir exists too
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
        saintq_df   = load_data(CONFIG["saintq_path"])
        saintex_df  = load_data(CONFIG["saintexpress_path"])
        limma_df    = load_data(CONFIG["limma_path"])        # <-- NEW
        gt_df       = load_data(CONFIG["plot_gt_file"])

        plot_pr_curve(
            features_df=features_df,
            saintq_df_raw=saintq_df,
            saintexpress_df_raw=saintex_df,
            limma_df_raw=limma_df,
            go_cc_df=gt_df,
            gt_source=CONFIG["plot_gt_source"],
            outdir=CONFIG["results_dir"],                    # <-- dataset-specific output root
        )

    elif args.step == "plot_recovery":
        features_df = load_data(CONFIG["feature_output"])
        saintq_df   = load_data(CONFIG["saintq_path"])
        saintex_df  = load_data(CONFIG["saintexpress_path"])
        limma_df    = load_data(CONFIG["limma_path"])
        gt_df       = load_data(CONFIG["plot_gt_file"])

        recovery_overlap(
            features_df=features_df,
            saintq_df_raw=saintq_df,
            saintexpress_df_raw=saintex_df,
            limma_df_raw=limma_df,
            go_cc_df=gt_df,
            gt_source=CONFIG["plot_gt_source"],
            outdir=CONFIG["results_dir"],
        )


    elif args.step == "plot_fdr_bins":
        features_df = load_data(CONFIG["feature_output"])
        saintq_df   = load_data(CONFIG["saintq_path"])
        saintex_df  = load_data(CONFIG["saintexpress_path"])
        limma_df    = load_data(CONFIG["limma_path"])
        gt_df       = load_data(CONFIG["plot_gt_file"])

        plot_fdr_bins(
            features_df=features_df,
            saintq_df_raw=saintq_df,
            saintexpress_df_raw=saintex_df,
            limma_df_raw=limma_df,
            go_cc_df=gt_df,
            gt_source=CONFIG["plot_gt_source"],
            outdir=CONFIG["results_dir"],
            background_exclusion=CONFIG.get("background_exclusion", True),
            background_flag_col=CONFIG.get("background_flag_col", "global_cv_flag"),
            background_flag_value=CONFIG.get("background_flag_value", "likely background"),
            fdr_bins=tuple(CONFIG.get("fdr_bins", [0.0, 0.01, 0.05])),
        )

    elif args.step == "plot_sensitivity":
        features_df = load_data(CONFIG["feature_output"])
        plot_sensitivity(
            features_df=features_df,
            outdir=CONFIG["results_dir"],
            initial_positives_list=CONFIG.get("initial_positives_grid", [1,5,10,15,20,25,30]),
            initial_negatives_list=CONFIG.get("initial_negatives_grid", [100,200,500,1000]),
            feature_columns=tuple(CONFIG.get("feature_columns", [
                "log_fold_change","snr","mean_diff","median_diff",
                "replicate_fold_change_sd","bait_cv","bait_control_sd_ratio","zero_or_neg_fc"
            ])),
            bait_col=CONFIG.get("bait_col","Bait"),
            prey_col=CONFIG.get("prey_col","Prey"),
            composite_col=CONFIG.get("composite_col","composite_score"),
            single_rep_flag_col=CONFIG.get("single_rep_flag_col","single_rep_flag"),
            prob_col=CONFIG.get("prob_col","predicted_probability"),
            fdr_col=CONFIG.get("fdr_col","FDR"),
            fdr_threshold_for_hits=CONFIG.get("sensitivity_fdr_threshold", 0.01),
            hist_bins_main=CONFIG.get("hist_bins_main", 100),
            hist_range=tuple(CONFIG.get("hist_range", [0,1])),
            seed=CONFIG.get("seed", 42),
            save_prefix=f"_{CONFIG['plot_gt_source']}" if "plot_gt_source" in CONFIG else ""
        )

    elif args.step == "plot_posneg_pr":
        features_df = load_data(CONFIG["feature_output"])
        gt_df = load_data(CONFIG["plot_gt_file"])
        plot_posneg_pr(
            features_df=features_df,
            gt_df=gt_df,
            outdir=CONFIG["results_dir"],
            gt_source=CONFIG["plot_gt_source"],
            positives_list=CONFIG.get("posneg_positives", [5,10,15,20,25,30]),
            negatives_list=CONFIG.get("posneg_negatives", [100,200,500,1000]),
            feature_columns=tuple(CONFIG.get("feature_columns", [
                "log_fold_change","snr","mean_diff","median_diff",
                "replicate_fold_change_sd","bait_cv","bait_control_sd_ratio","zero_or_neg_fc"
            ])),
            bait_col=CONFIG.get("bait_col","Bait"),
            prey_col=CONFIG.get("prey_col","Prey"),
            composite_col=CONFIG.get("composite_col","composite_score"),
            single_rep_flag_col=CONFIG.get("single_rep_flag_col","single_rep_flag"),
            n_bootstrap=CONFIG.get("n_bootstrap", 100),
            topn_for_auc=CONFIG.get("topn_for_auc", 300),
            seed=CONFIG.get("seed", 42),
            cache_csv=CONFIG.get("cache_csv", True),
        )

    elif args.step == "plot_fdr_robustness":
        # Use trained WHISPER output that contains probabilities & FDR
        features_df = load_data(CONFIG["fdr_output"])
        res = plot_fdr_robustness(
            features_df=features_df,
            outdir=CONFIG["results_dir"],
            label_pred_col=CONFIG.get("label_pred_col", "predicted_probability"),
            fdr_col=CONFIG.get("fdr_col", "FDR"),
            mean_diff_col=CONFIG.get("mean_diff_col", "mean_diff"),
            bait_col=CONFIG.get("bait_col", "Bait"),
            prey_col=CONFIG.get("prey_col", "Prey"),
            do_alternative_nulls=CONFIG.get("do_alternative_nulls", True),
            hist_bins_main=CONFIG.get("hist_bins_main", 40),
            hist_bins_summary=CONFIG.get("hist_bins_summary", 10),
            log_ylim_pad=CONFIG.get("log_ylim_pad", 10.0),
            seed=CONFIG.get("seed", 42),
            save_prefix=f"_{CONFIG['plot_gt_source']}" if "plot_gt_source" in CONFIG else ""
        )


    elif args.step == "plot_go_upset":
        # Use trained WHISPER output + external methods
        puppi_df = load_data(CONFIG["fdr_output"])            # WHISPER results
        saintq_df = load_data(CONFIG["saintq_path"])
        saintex_df = load_data(CONFIG["saintexpress_path"])
        limma_df = load_data(CONFIG["limma_path"])

        # Dump everything directly into the dataset-specific results_dir
        outdir = CONFIG["results_dir"]
        fdr_thr = CONFIG.get("go_fdr_threshold", 0.01)
        go_user_thr = CONFIG.get("go_user_threshold", 0.05)

        run_go_upset_analysis(
            puppi_df=puppi_df,
            limma_df=limma_df,
            saintq_df=saintq_df,
            saintexpress_df=saintex_df,
            outdir=outdir,
            fdr_threshold=fdr_thr,
            go_user_threshold=go_user_thr,
        )

    elif args.step == "plot_hpa":
        whisper_df = load_data(CONFIG["fdr_output"])
        saintq_df = load_data(CONFIG["saintq_path"])
        saintex_df = load_data(CONFIG["saintexpress_path"])
        limma_df = load_data(CONFIG["limma_path"])

        plot_hpa_annotations(
            whisper_df=whisper_df,
            saintq_df=saintq_df,
            saintexpress_df=saintex_df,
            limma_df=limma_df,
            hpa_tsv=CONFIG["hpa_path"],
            bait=CONFIG["hpa_bait"],
            outdir=CONFIG["results_dir"],
            fdr_threshold=CONFIG.get("fdr_threshold", 0.01),
        )

    elif args.step == "full":
        df = load_data(CONFIG["input_file"])
        features_df = run_feature_engineering(df, controls=CONFIG["controls"])
        save_csv(features_df, CONFIG["feature_output"])
        final_df = run_training_and_fdr(features_df, CONFIG["initial_positives"], CONFIG["initial_negatives"])
        save_csv(final_df, CONFIG["fdr_output"])

if __name__ == "__main__":
    main()