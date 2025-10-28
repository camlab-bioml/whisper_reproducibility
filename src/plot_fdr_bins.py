# src/plot_fdr_bins.py

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
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def plot_fdr_bins(
    features_df: pd.DataFrame,
    saintq_df_raw: pd.DataFrame,
    saintexpress_df_raw: pd.DataFrame,
    limma_df_raw: pd.DataFrame,
    go_cc_df: pd.DataFrame,
    gt_source: str = "go_large",
    outdir: str = "results/Benchmarking BioID DIA",
    background_exclusion: bool = True,
    background_flag_col: str = "global_cv_flag",
    background_flag_value: str = "likely background",
    fdr_bins=(0.0, 0.01, 0.05),
):
    """
    Precision & TP counts by FDR bin (aggregated across all baits), comparing:
      - whisper (WHISPER) [FDR column: 'FDR']
      - SAINTq & SAINTexpress [FDR/BFDR column: 'BFDR']
      - limma [adjusted p-value column: 'adj.P.Val']

    Parameters
    ----------
    features_df : DataFrame
        Must include columns: ['Bait', 'Prey', 'FDR'] and (optionally) background_flag_col.
    saintq_df_raw : DataFrame
        Must include columns: ['Bait','PreyGene','BFDR','AvgP'] (AvgP unused here, but common schema).
    saintexpress_df_raw : DataFrame
        Must include columns: ['Bait','PreyGene','BFDR','AvgP'].
    limma_df_raw : DataFrame
        Must include columns: ['Bait','Protein','adj.P.Val'] (will be renamed to Prey).
    go_cc_df : DataFrame
        Ground truth with columns: ['Bait','Prey'] (positives).
    gt_source : str
        Label used in output filenames.
    outdir : str
        Directory to save figures and CSV outputs.
    background_exclusion : bool
        If True, also compute "PU excl. background" by removing rows where background_flag_col == background_flag_value.
    background_flag_col : str
        Column name used to flag likely-background rows in features_df.
    background_flag_value : str
        Value in background_flag_col indicating likely background.
    fdr_bins : tuple
        Bin edges for FDR (e.g., (0.0, 0.01, 0.05)).

    Outputs
    -------
    - CSV: precision_tp_results_by_bin_{gt_source}[...].csv
    - PDF: tp_precision_barplot_{gt_source}[...].pdf
    """

    os.makedirs(outdir, exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    fdr_bin_labels = [f"{fdr_bins[i]}-{fdr_bins[i+1]}" for i in range(len(fdr_bins) - 1)]

    def _add_missing_preys(df_in: pd.DataFrame, prey_col: str) -> pd.DataFrame:
        """
        For each bait, ensure all preys in features_df are present with default worst FDR.
        Avoids duplication by building per-bait then concatenating once.
        """
        out_chunks = []
        features_by_bait = features_df.groupby("Bait")
        for bait, grp in features_by_bait:
            unique_preys = set(grp["Prey"])
            sub = df_in[df_in["Bait"] == bait].copy()
            existing_preys = set(sub[prey_col]) if not sub.empty else set()
            missing = list(unique_preys - existing_preys)
            if len(missing) > 0:
                if prey_col == "PreyGene":
                    miss_df = pd.DataFrame(
                        {"Bait": bait, "PreyGene": missing, "BFDR": 1.0, "AvgP": 0.0}
                    )
                else:
                    # limma case (prey_col == 'Prey')
                    miss_df = pd.DataFrame(
                        {"Bait": bait, "Prey": missing, "adj.P.Val": 1.0}
                    )
                sub = pd.concat([sub, miss_df], ignore_index=True)
            out_chunks.append(sub)
        return pd.concat(out_chunks, ignore_index=True) if out_chunks else df_in.copy()

    def _compute_precision_and_tp(df, fdr_column, bins, drop_bg=False):
        tmp = df.copy()
        if drop_bg and background_flag_col in tmp.columns:
            tmp = tmp[tmp[background_flag_col] != background_flag_value]
        tmp["FDR_bin"] = pd.cut(tmp[fdr_column], bins=bins, labels=fdr_bin_labels, include_lowest=True)
        # For each bin: TP, Total, Precision
        out = (
            tmp.groupby("FDR_bin")["true_label"]
            .agg(TP="sum", Total="count")
            .reset_index()
        )
        out["Precision"] = out["TP"] / out["Total"].replace(0, np.nan)
        out["Precision"] = out["Precision"].fillna(0.0)
        return out  # columns: FDR_bin, TP, Total, Precision

    # -----------------------------
    # Prepare inputs (schemas)
    # -----------------------------
    # SAINT tables: ensure complete prey coverage
    saintq_df = saintq_df_raw.copy()
    saintex_df = saintexpress_df_raw.copy()
    limma_df = limma_df_raw.copy()

    # limma standardize prey column
    if "Protein" in limma_df.columns and "Prey" not in limma_df.columns:
        limma_df = limma_df.rename(columns={"Protein": "Prey"})

    saintq_df = _add_missing_preys(saintq_df, "PreyGene")
    saintex_df = _add_missing_preys(saintex_df, "PreyGene")
    limma_df = _add_missing_preys(limma_df, "Prey")

    # -----------------------------
    # Ground truth dict by bait
    # -----------------------------
    # Restrict GT to identified preys
    identified_preys = set(features_df["Prey"].unique())
    go_cc_df = go_cc_df[go_cc_df["Prey"].isin(identified_preys)].copy()
    gt_dict = {b: set(df["Prey"]) for b, df in go_cc_df.groupby("Bait")}

    # -----------------------------
    # Per-bait processing
    # -----------------------------
    all_bait_rows = []
    methods = ["WHISPER", "SAINTq", "SAINTexpress", "limma"]
    if background_exclusion:
        methods.insert(1, "PU excl. background")

    # cumulative aggregations across baits
    cumulative = {
        m: {lab: {"TP": 0, "Total": 0} for lab in fdr_bin_labels} for m in methods
    }

    for bait in features_df["Bait"].unique():
        gold = gt_dict.get(bait, set())

        # whisper/PU (features_df)
        bait_pu = features_df[features_df["Bait"] == bait].copy()
        bait_pu["true_label"] = bait_pu["Prey"].isin(gold).astype(int)

        # SAINTq / SAINTexpress (PreyGene)
        bait_q = saintq_df[saintq_df["Bait"] == bait].copy()
        bait_q["true_label"] = bait_q["PreyGene"].isin(gold).astype(int)

        bait_ex = saintex_df[saintex_df["Bait"] == bait].copy()
        bait_ex["true_label"] = bait_ex["PreyGene"].isin(gold).astype(int)

        # limma (Prey)
        bait_lim = limma_df[limma_df["Bait"] == bait].copy()
        bait_lim["true_label"] = bait_lim["Prey"].isin(gold).astype(int)

        # Compute per-bin stats
        pu_incl = _compute_precision_and_tp(bait_pu, "FDR", fdr_bins, drop_bg=False)
        if background_exclusion:
            pu_excl = _compute_precision_and_tp(bait_pu, "FDR", fdr_bins, drop_bg=True)
        q_stats = _compute_precision_and_tp(bait_q, "BFDR", fdr_bins, drop_bg=False)
        ex_stats = _compute_precision_and_tp(bait_ex, "BFDR", fdr_bins, drop_bg=False)
        lim_stats = _compute_precision_and_tp(bait_lim, "adj.P.Val", fdr_bins, drop_bg=False)

        # Collect rows & update cumulative
        def _collect(method_name, df_stats):
            for _, r in df_stats.iterrows():
                all_bait_rows.append(
                    {
                        "Bait": bait,
                        "FDR Bin": r["FDR_bin"],
                        "Method": method_name,
                        "Precision": float(r["Precision"]),
                        "TP": int(r["TP"]),
                        "Total": int(r["Total"]),
                    }
                )
                cumulative[method_name][r["FDR_bin"]]["TP"] += int(r["TP"])
                cumulative[method_name][r["FDR_bin"]]["Total"] += int(r["Total"])

        _collect("WHISPER", pu_incl)
        if background_exclusion:
            _collect("PU excl. background", pu_excl)
        _collect("SAINTq", q_stats)
        _collect("SAINTexpress", ex_stats)
        _collect("limma", lim_stats)

    results_df = pd.DataFrame(all_bait_rows)

    # -----------------------------
    # Save CSV
    # -----------------------------
    suffix = "_background_exclusion" if background_exclusion else ""
    csv_path = os.path.join(outdir, f"precision_tp_results_by_bin_{gt_source}{suffix}.csv")
    results_df.to_csv(csv_path, index=False)

    # -----------------------------
    # Aggregate for visualization
    # -----------------------------
    plot_rows = []
    for m in methods:
        for b in fdr_bin_labels:
            tp = cumulative[m][b]["TP"]
            total = cumulative[m][b]["Total"]
            prec = (tp / total) if total > 0 else 0.0
            plot_rows.append({"Method": m, "FDR Bin": b, "TP": tp, "Precision": prec})
    plot_df = pd.DataFrame(plot_rows)

    # -----------------------------
    # Plot
    # -----------------------------
    colors = {
        "WHISPER": "#003f5c",
        "PU excl. background": "#9ecae9",
        "SAINTq": "#ffb347",
        "SAINTexpress": "#ff7f0e",
        "limma": "#d62728",
    }

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 14), sharex=True)

    # Top: TP
    ax_top = axes[0]
    sns.barplot(
        data=plot_df,
        x="FDR Bin",
        y="TP",
        hue="Method",
        palette=colors,
        order=fdr_bin_labels,
        ax=ax_top,
    )
    ax_top.set_ylabel("True Positives", fontsize=26)
    ax_top.set_xlabel("", fontsize=26)
    ax_top.tick_params(axis="x", labelsize=24)
    ax_top.tick_params(axis="y", labelsize=24)
    if ax_top.legend_:
        ax_top.legend_.remove()

    # Bottom: Precision
    ax_bot = axes[1]
    sns.barplot(
        data=plot_df,
        x="FDR Bin",
        y="Precision",
        hue="Method",
        palette=colors,
        order=fdr_bin_labels,
        ax=ax_bot,
    )
    ax_bot.set_ylabel("Precision", fontsize=26)
    ax_bot.set_xlabel("FDR Bin", fontsize=26)
    ax_bot.tick_params(axis="x", labelsize=24)
    ax_bot.tick_params(axis="y", labelsize=24)
    if ax_bot.legend_:
        ax_bot.legend_.remove()

    # Shared legend
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=18,
        title_fontsize=20,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    pdf_path = os.path.join(outdir, f"tp_precision_barplot_{gt_source}{suffix}.pdf")
    plt.savefig(pdf_path, dpi=300)
    plt.close()

    return {"csv": csv_path, "pdf": pdf_path}
