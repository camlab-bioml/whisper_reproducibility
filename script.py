import argparse
import os
import pandas as pd
from src.feature_engineering import run_feature_engineering
from src.train_and_fdr import run_training_and_fdr

def load_data(input_file):
    print(f"[INFO] Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Run PU Learning Pipeline")
    parser.add_argument("--step", choices=["feature_engineering", "train_and_fdr", "full"], required=True)

    parser.add_argument("--input_file", help="Input intensity data (TSV or CSV)", required=True)
    parser.add_argument("--feature_output", help="Path to save feature CSV", required=True)
    parser.add_argument("--fdr_output", help="Path to save final FDR + probability CSV", required=True)

    parser.add_argument("--control_keywords", nargs="+", required=True,
                        help="List of control keywords (e.g. EGFP Empty NminiTurbo)")

    parser.add_argument("--initial_positives", type=int, default=10, help="Number of initial positives per bait")
    parser.add_argument("--initial_negatives", type=int, default=200, help="Number of negatives per bait")

    args = parser.parse_args()

    if args.step == "feature_engineering":
        df = load_data(args.input_file)
        features_df = run_feature_engineering(df, control_keywords=args.control_keywords)
        save_csv(features_df, args.feature_output)

    elif args.step == "train_and_fdr":
        features_df = load_data(args.feature_output)
        final_df = run_training_and_fdr(features_df, initial_positives=args.initial_positives,
                                        initial_negatives=args.initial_negatives)
        save_csv(final_df, args.fdr_output)

    elif args.step == "full":
        df = load_data(args.input_file)
        features_df = run_feature_engineering(df, control_keywords=args.control_keywords)
        save_csv(features_df, args.feature_output)

        final_df = run_training_and_fdr(features_df, initial_positives=args.initial_positives,
                                        initial_negatives=args.initial_negatives)
        save_csv(final_df, args.fdr_output)


if __name__ == "__main__":
    main()
