# src/plot_fdr_robustness.py

# ---------------- Matplotlib defaults ------------------------
import matplotlib
matplotlib.rcParams["pdf.fonttype"]    = 42
matplotlib.rcParams["ps.fonttype"]     = 42
matplotlib.rcParams["figure.dpi"]      = 300
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def plot_fdr_robustness(
    features_df: pd.DataFrame,
    outdir: str = "results/Benchmarking BioID DIA",
    label_pred_col: str = "predicted_probability",
    fdr_col: str = "FDR",
    mean_diff_col: str = "mean_diff",
    bait_col: str = "Bait",
    prey_col: str = "Prey",
    do_alternative_nulls: bool = True,
    hist_bins_main: int = 40,
    hist_bins_summary: int = 10,
    log_ylim_pad: float = 10.0,
    seed: int = 42,
    save_prefix: str = ""
):
    """
    Make FDR robustness plots:
      1) Compare the distribution of whisper FDRs vs. two alternative null-based FDRs for `mean_diff`:
         - value shuffle within bait
         - random bait–prey re-pairing
      2) Histogram panels (log-scaled) and annotated bar plots.

    Assumptions
    ----------
    features_df:
        Must contain at least:
          - columns: [bait_col, prey_col, mean_diff_col]
          - a model probability column (label_pred_col) and FDR column (fdr_col)
        Typically produced by your `train_and_fdr` step.

    Outputs
    -------
    - CSV: mean_diff_fdr_all_methods{save_prefix}.csv
    - PDF: compare_fdr_distributions{save_prefix}.pdf
    - PDF: fdr_histogram_comparison_all_methods{save_prefix}.pdf
    - PDF: fdr_histogram_comparison_annotated{save_prefix}.pdf
    """

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Basic checks
    for c in [bait_col, prey_col, mean_diff_col]:
        if c not in features_df.columns:
            raise ValueError(f"`features_df` is missing required column '{c}'")

    if label_pred_col not in features_df.columns or fdr_col not in features_df.columns:
        raise ValueError(
            f"`features_df` must include '{label_pred_col}' and '{fdr_col}' (from train_and_fdr step)."
        )

    df = features_df.copy().sort_values([bait_col, prey_col]).reset_index(drop=True)

    # -----------------------------
    # Alternative null FDRs for `mean_diff`
    # -----------------------------
    if do_alternative_nulls:
        # (A1) Value shuffle within bait (null preserves bait distribution, breaks prey mapping)
        decoy_md_scores = []
        for bait, grp in df.groupby(bait_col):
            vals = grp[mean_diff_col].dropna().values
            if len(vals) > 0:
                shuffled = vals.copy()
                rng.shuffle(shuffled)
                decoy_md_scores.extend(shuffled)
        decoy_md_scores = np.asarray(decoy_md_scores)

        # (A2) Random bait–prey re-pairing (null breaks bait–prey pairing)
        baits_all = df[bait_col].values
        preys_all = df[prey_col].values
        decoy_baits = rng.choice(baits_all, size=len(df), replace=True)
        decoy_preys = rng.choice(preys_all, size=len(df), replace=True)
        df_pairs = pd.DataFrame({bait_col: decoy_baits, prey_col: decoy_preys})
        df_pairs = df_pairs.merge(
            df[[bait_col, prey_col, mean_diff_col]], on=[bait_col, prey_col], how="left"
        )
        decoy_md_pair_scores = df_pairs[mean_diff_col].dropna().values

        # FDR curves per unique real score (monotone-free by construction here)
        real_md_scores = df[mean_diff_col].values
        unique_md_real = np.unique(real_md_scores)

        fdr_md1_dict, fdr_md2_dict = {}, {}
        for score in unique_md_real:
            num_real = np.sum(real_md_scores >= score)
            if num_real == 0:
                fdr_md1_dict[score] = 1.0
                fdr_md2_dict[score] = 1.0
                continue
            num_decoy_1 = np.sum(decoy_md_scores >= score)
            num_decoy_2 = np.sum(decoy_md_pair_scores >= score)
            fdr_md1_dict[score] = min(num_decoy_1 / num_real, 1.0)
            fdr_md2_dict[score] = min(num_decoy_2 / num_real, 1.0)

        df["mean_diff_FDR_shuffle"] = df[mean_diff_col].map(fdr_md1_dict)
        df["mean_diff_FDR_pairs"] = df[mean_diff_col].map(fdr_md2_dict)
    else:
        # If not computing, fill with NaN so plotting can proceed with what exists
        if "mean_diff_FDR_shuffle" not in df.columns:
            df["mean_diff_FDR_shuffle"] = np.nan
        if "mean_diff_FDR_pairs" not in df.columns:
            df["mean_diff_FDR_pairs"] = np.nan

    # -----------------------------
    # Save CSV of FDRs for record
    # -----------------------------
    csv_out = os.path.join(
        outdir, f"mean_diff_fdr_all_methods{save_prefix}.csv"
    )
    df[[bait_col, prey_col, mean_diff_col, "mean_diff_FDR_shuffle", "mean_diff_FDR_pairs", fdr_col]].to_csv(
        csv_out, index=False
    )

    # -----------------------------
    # Plot 1: Overlaid distributions (hist + KDE)
    # -----------------------------
    plt.figure(figsize=(12, 5))

    # (left) hist
    plt.subplot(1, 2, 1)
    lab_map = {
        fdr_col: "whisper (PU)",
        "mean_diff_FDR_shuffle": "mean_diff (value shuffle)",
        "mean_diff_FDR_pairs": "mean_diff (bait–prey shuffle)",
    }
    colors = {
        fdr_col: "blue",
        "mean_diff_FDR_shuffle": "orange",
        "mean_diff_FDR_pairs": "green",
    }
    for col in [fdr_col, "mean_diff_FDR_shuffle", "mean_diff_FDR_pairs"]:
        if df[col].notna().any():
            plt.hist(
                df[col].dropna().values,
                bins=hist_bins_main,
                alpha=0.6,
                label=lab_map[col],
                color=colors[col],
            )
    plt.xlabel("Estimated FDR")
    plt.ylabel("Number of interactions")
    plt.title("Histogram of Assigned FDRs")
    plt.legend()

    # (right) KDE
    plt.subplot(1, 2, 2)
    for col in [fdr_col, "mean_diff_FDR_shuffle", "mean_diff_FDR_pairs"]:
        if df[col].notna().any():
            sns.kdeplot(df[col].dropna().values, label=lab_map[col], color=colors[col])
    plt.xlabel("Estimated FDR")
    plt.ylabel("Density")
    plt.title("Density of Assigned FDRs (log y)")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    pdf_compare = os.path.join(outdir, f"compare_fdr_distributions{save_prefix}.pdf")
    plt.savefig(pdf_compare, dpi=300)
    plt.close()

    # -----------------------------
    # Plot 2: Three-panel histograms on shared log scale
    # -----------------------------
    # compute a shared ymax across panels
    def _hist_counts(col, bins):
        vals = df[col].dropna().values
        if len(vals) == 0:
            return np.array([0])
        return np.histogram(vals, bins=bins)[0]

    counts_pu  = _hist_counts(fdr_col, hist_bins_summary)
    counts_md1 = _hist_counts("mean_diff_FDR_shuffle", hist_bins_main)
    counts_md2 = _hist_counts("mean_diff_FDR_pairs", hist_bins_summary)
    max_val = max(int(counts_pu.max() if counts_pu.size else 1),
                  int(counts_md1.max() if counts_md1.size else 1),
                  int(counts_md2.max() if counts_md2.size else 1))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if df[fdr_col].notna().any():
        plt.hist(df[fdr_col].dropna().values, bins=hist_bins_summary, color="blue", alpha=0.85)
    plt.xlabel("Estimated FDR")
    plt.ylabel("Number of interactions")
    plt.yscale("log")
    plt.ylim(1e-1, max_val * log_ylim_pad)
    plt.title("whisper (PU)")

    plt.subplot(1, 3, 2)
    if df["mean_diff_FDR_shuffle"].notna().any():
        plt.hist(df["mean_diff_FDR_shuffle"].dropna().values, bins=hist_bins_main + 1, color="orange", alpha=0.85)
    plt.xlabel("Estimated FDR")
    plt.yscale("log")
    plt.ylim(1e-1, max_val * log_ylim_pad)
    plt.title("mean_diff (value shuffle)")

    plt.subplot(1, 3, 3)
    if df["mean_diff_FDR_pairs"].notna().any():
        plt.hist(df["mean_diff_FDR_pairs"].dropna().values, bins=hist_bins_summary, color="green", alpha=0.85)
    plt.xlabel("Estimated FDR")
    plt.yscale("log")
    plt.ylim(1e-1, max_val * log_ylim_pad)
    plt.title("mean_diff (bait–prey shuffle)")

    plt.tight_layout()
    pdf_hist = os.path.join(outdir, f"fdr_histogram_comparison_all_methods{save_prefix}.pdf")
    plt.savefig(pdf_hist, dpi=300)
    plt.close()

    # -----------------------------
    # Plot 3: Annotated bar histograms on shared log scale
    # -----------------------------
    # Precompute histograms to annotate
    def _histogram(col, bins):
        vals = df[col].dropna().values
        if len(vals) == 0:
            return (np.zeros(bins if isinstance(bins, int) else len(bins)-1, dtype=int), 
                    np.linspace(0, 1, (bins if isinstance(bins, int) else len(bins)-1) + 1))
        return np.histogram(vals, bins=bins)

    h1, b1 = _histogram(fdr_col, hist_bins_summary)
    h2, b2 = _histogram("mean_diff_FDR_shuffle", hist_bins_main)
    h3, b3 = _histogram("mean_diff_FDR_pairs", hist_bins_summary)
    max_val2 = max(int(h1.max() if h1.size else 1),
                   int(h2.max() if h2.size else 1),
                   int(h3.max() if h3.size else 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (axes[0], h1, b1, "blue",  "whisper (PU)"),
        (axes[1], h2, b2, "orange","mean_diff (value shuffle)"),
        (axes[2], h3, b3, "green", "mean_diff (bait–prey shuffle)"),
    ]
    for ax, hist, bins, color, title in panels:
        if hist.size:
            ax.bar(bins[:-1], hist, width=np.diff(bins), align="center", color=color, alpha=0.85)
            # annotate
            for count, x, w in zip(hist, bins[:-1], np.diff(bins)):
                if count > 0:
                    ax.text(x + w*0.5, count * 1.15, str(int(count)), fontsize=7, rotation=90, ha="center")
        ax.set_yscale("log")
        ax.set_ylim(1e-1, max_val2 * log_ylim_pad)
        ax.set_xlabel("Estimated FDR")
        ax.set_title(title)
    axes[0].set_ylabel("Number of interactions")

    plt.tight_layout()
    pdf_annot = os.path.join(outdir, f"fdr_histogram_comparison_annotated{save_prefix}.pdf")
    plt.savefig(pdf_annot, dpi=300)
    plt.close()

    return {
        "csv": csv_out,
        "pdf_compare": pdf_compare,
        "pdf_hist": pdf_hist,
        "pdf_annot": pdf_annot,
    }
