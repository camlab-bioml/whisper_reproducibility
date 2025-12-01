# src/plot_go_upset.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from upsetplot import from_memberships, UpSet
from gprofiler import GProfiler
from matplotlib import cm
from matplotlib.colors import Normalize

import matplotlib
matplotlib.rcParams["pdf.fonttype"]    = 42
matplotlib.rcParams["ps.fonttype"]     = 42
matplotlib.rcParams["figure.dpi"]      = 300
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return cm.colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )


def get_sig_preys(df, bait, method, fdr_threshold):
    """Return set of significant preys for a bait for each method."""
    if method == "limma":
        filtered_df = df[(df["Bait"] == bait) & (df["adj.P.Val"] <= fdr_threshold)]
        preys = filtered_df["Prey"].dropna()
        return set(preys.astype(str))

    elif method in ["SAINTq", "SAINTexpress"]:
        filtered_df = df[(df["Bait"] == bait) & (df["BFDR"] <= fdr_threshold)]
        preys = filtered_df["PreyGene"].dropna()
        return set(preys.astype(str))

    else:  # WHISPER / puppi
        filtered_df = df[(df["Bait"] == bait) & (df["FDR"] <= fdr_threshold)]
        preys = filtered_df["Prey"].dropna()
        return set(preys.astype(str))


def run_gprofiler(preys, bait, method, gp, user_threshold=0.05):
    if len(preys) < 3:
        return None
    try:
        res = gp.profile(
            organism="hsapiens",
            query=list(preys),
            sources=["GO:CC"],
            user_threshold=user_threshold,
        )
        res["Bait"] = bait
        res["Method"] = method
        return res
    except Exception:
        return None


def plot_gprofiler_dotplot(
    enrich_df,
    bait,
    outdir,
    top_terms=10,
    max_term_length=40,
    max_term_size=2500,
):
    """
    Make a GO:CC dot plot for a single bait (overall or UpSet-based),
    saving directly into `outdir` as:
        {bait}_gprofiler_dotplot.pdf
    """
    os.makedirs(outdir, exist_ok=True)
    df = enrich_df.copy()

    # Filter by term size
    if "term_size" in df.columns:
        df = df[df["term_size"] <= max_term_size]

    # Truncate long term names for x-axis
    df["name"] = df["name"].apply(
        lambda x: x[:max_term_length] + "..." if isinstance(x, str) and len(x) > max_term_length else x
    )

    # Take top-N by p-value per method
    filtered_df = (
        df.groupby("Method", group_keys=False)
          .apply(lambda x: x.nsmallest(top_terms, "p_value"))
          .reset_index(drop=True)
    )
    if filtered_df.empty:
        return

    # Color scale: -log10(p)
    filtered_df["neglog10_p"] = -np.log10(
        filtered_df["p_value"].replace(0, np.nextafter(0, 1))
    )

    # Point sizes based on intersection_size
    S_MIN, S_MAX = 50, 300
    vmin = float(filtered_df["intersection_size"].min())
    vmax = float(filtered_df["intersection_size"].max())
    if vmax <= 0:
        vmax = 1.0
    if vmin < 0:
        vmin = 0.0

    # Figure size
    n_terms   = filtered_df["name"].nunique()
    n_methods = filtered_df["Method"].nunique()
    fig_w = 10
    fig_h = 4

    cmap = truncate_colormap(cm.get_cmap("Oranges"), 0.35, 1.0)

    # Map methods to numeric positions on y axis
    method_order = sorted(filtered_df["Method"].unique())
    method_map = {m: i for i, m in enumerate(method_order)}
    filtered_df["Method_pos"] = filtered_df["Method"].map(method_map)

    fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    sns.scatterplot(
        data=filtered_df,
        x="name",
        y="Method_pos",
        size="intersection_size",
        hue="neglog10_p",
        sizes=(S_MIN, S_MAX),
        palette=cmap,
        ax=ax1,
        edgecolor="none",
    )

    ax1.set_xlabel("GO Term")
    ax1.set_ylabel("Method")
    ax1.set_title(f"Enriched GO Terms for {bait}", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    # Show method names on y axis
    ax1.set_yticks(range(len(method_order)))
    ax1.set_yticklabels(method_order)
    ax1.set_ylim(-0.5, len(method_order) - 0.5)

    # Remove combined legend
    if ax1.legend_:
        ax1.legend_.remove()

    # Custom legend: intersection size (min / max)
    def area_from_val(v):
        if vmax == vmin:
            return S_MAX
        return S_MIN + (float(v) - vmin) / (vmax - vmin) * (S_MAX - S_MIN)

    min_ms = np.sqrt(area_from_val(vmin))
    max_ms = np.sqrt(area_from_val(vmax))
    size_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markersize=min_ms, label=f"{int(vmin)}"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markersize=max_ms, label=f"{int(vmax)}"),
    ]
    ax1.legend(
        handles=size_handles,
        title="Intersection Size",
        loc="upper left",
        bbox_to_anchor=(1.1, 1.0),
        frameon=True,
    )

    # Colorbar: -log10(p)
    plt.subplots_adjust(right=0.86)
    norm = Normalize(
        vmin=float(filtered_df["neglog10_p"].min()),
        vmax=float(filtered_df["neglog10_p"].max()),
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.01)
    cbar.set_label(r"$-\log_{10}$ (Adjusted P-Value)", fontsize=10)

    # Save directly into outdir
    out_path = os.path.join(outdir, f"{bait}_gprofiler_dotplot.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GO/UpSet] Saved dot plot: {out_path}")


def plot_upset(preys_dict, bait, outdir, fdr_threshold):
    """
    Make an UpSet plot of overlaps across methods for a single bait.
    Saves directly into `outdir` as:
        {bait}_upset.pdf
    """
    os.makedirs(outdir, exist_ok=True)

    all_preys = set.union(*preys_dict.values())
    memberships = []

    for prey in all_preys:
        methods_present = [
            method for method, preys in preys_dict.items() if prey in preys
        ]
        if methods_present:
            memberships.append(tuple(methods_present))

    if not memberships:
        return None, None

    data = from_memberships(memberships)

    plt.figure(figsize=(4, 10))
    upset = UpSet(data, subset_size="count", show_counts=True, sort_by="degree")
    upset.plot()
    plt.suptitle(
        f"{bait}: Significant Prey Overlaps (FDR ≤ {fdr_threshold:.2f})",
        fontsize=14,
    )

    out_path = os.path.join(outdir, f"{bait}_upset.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GO/UpSet] Saved UpSet plot: {out_path}")

    return data, memberships


def run_go_upset_analysis(
    puppi_df,
    limma_df,
    saintq_df,
    saintexpress_df,
    outdir,
    fdr_threshold=0.01,
    go_user_threshold=0.05,
):
    """
    High-level wrapper: for each bait, run GO:CC enrichment (overall + UpSet groups)
    and generate dot plots + UpSet plots + Excel tables.

    All outputs are saved directly inside `outdir`:
      - {bait}_gprofiler_overall.xlsx
      - {bait}_gprofiler_upset_enrichment.xlsx
      - {bait}_gprofiler_upset_preys.xlsx
      - {bait}_overall_gprofiler_dotplot.pdf
      - {bait}_upset_gprofiler_dotplot.pdf
      - {bait}_upset.pdf
    """

    os.makedirs(outdir, exist_ok=True)

    # unify limma column
    if "Protein" in limma_df.columns and "Prey" not in limma_df.columns:
        limma_df = limma_df.rename(columns={"Protein": "Prey"})

    all_dfs = {
        "puppi": puppi_df,
        "limma": limma_df,
        "SAINTq": saintq_df,
        "SAINTexpress": saintexpress_df,
    }

    baits = puppi_df["Bait"].unique()
    gp = GProfiler(return_dataframe=True)

    for bait in baits:
        print(f"\n=== [GO/UpSet] Processing bait: {bait} ===")

        # 1) significant preys for each method
        preys_dict = {
            m: get_sig_preys(df, bait, m, fdr_threshold)
            for m, df in all_dfs.items()
        }

        # 2) Overall enrichment per method
        bait_enrich = {}
        for method, preys in preys_dict.items():
            res = run_gprofiler(
                preys,
                bait,
                method,
                gp,
                user_threshold=go_user_threshold,
            )
            if res is not None and not res.empty:
                bait_enrich[method] = res

        if bait_enrich:
            # overall enrichment table (one Excel per bait)
            overall_path = os.path.join(
                outdir,
                f"{bait}_gprofiler_overall.xlsx",
            )
            with pd.ExcelWriter(overall_path) as writer:
                for method, df in bait_enrich.items():
                    df.to_excel(writer, sheet_name=method, index=False)
            print(f"[GO/UpSet] Saved overall enrichment table: {overall_path}")

            full_df = pd.concat(bait_enrich.values(), ignore_index=True)
            plot_gprofiler_dotplot(
                enrich_df=full_df,
                bait=f"{bait}_overall",
                outdir=outdir,
                top_terms=10,
            )

        # 3) UpSet plot
        upset_data, memberships = plot_upset(
            preys_dict, bait, outdir, fdr_threshold
        )
        upset_enrich = {}

        # 4) Enrichment per UpSet combination
        if upset_data is not None and memberships is not None:
            upset_path = os.path.join(
                outdir,
                f"{bait}_gprofiler_upset_enrichment.xlsx",
            )
            preys_path = os.path.join(
                outdir,
                f"{bait}_gprofiler_upset_preys.xlsx",
            )

            with pd.ExcelWriter(upset_path) as writer, pd.ExcelWriter(
                preys_path
            ) as preys_writer:

                unique_combos = list(set(memberships))

                for methods_tuple in unique_combos:
                    if not methods_tuple:
                        continue

                    methods_in_combo = list(methods_tuple)
                    all_preys_union = set.union(*preys_dict.values())
                    exclusive_preys = set()

                    for prey in all_preys_union:
                        prey_methods = {
                            method
                            for method, preys in preys_dict.items()
                            if prey in preys
                        }
                        if prey_methods == set(methods_in_combo):
                            exclusive_preys.add(prey)

                    combo_name = "+".join(methods_in_combo)
                    print(
                        f"  UpSet combo {combo_name}: {len(exclusive_preys)} exclusive preys"
                    )

                    # Save preys for this combo
                    prey_df = pd.DataFrame(
                        sorted(exclusive_preys), columns=["Prey"]
                    )
                    sheet_name_preys = combo_name
                    if len(sheet_name_preys) > 31:
                        sheet_name_preys = sheet_name_preys[:28] + "..."
                    prey_df.to_excel(
                        preys_writer, sheet_name=sheet_name_preys, index=False
                    )

                    # GO enrichment for this combo
                    res = run_gprofiler(
                        exclusive_preys,
                        bait,
                        combo_name,
                        gp,
                        user_threshold=go_user_threshold,
                    )

                    sheet_name = combo_name
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:28] + "..."

                    if res is not None and not res.empty:
                        res.to_excel(writer, sheet_name=sheet_name, index=False)
                        upset_enrich[combo_name] = res
                    else:
                        empty_df = pd.DataFrame(
                            columns=[
                                "source",
                                "native",
                                "name",
                                "description",
                                "p_value",
                                "significant",
                                "term_size",
                                "query_size",
                                "intersection_size",
                                "intersection",
                                "effective_domain_size",
                                "precision",
                                "recall",
                                "Bait",
                                "Method",
                            ]
                        )
                        empty_df["Bait"] = bait
                        empty_df["Method"] = combo_name
                        empty_df.to_excel(
                            writer, sheet_name=sheet_name, index=False
                        )

            if upset_enrich:
                upset_df = pd.concat(upset_enrich.values(), ignore_index=True)
                plot_gprofiler_dotplot(
                    enrich_df=upset_df,
                    bait=f"{bait}_upset",
                    outdir=outdir,
                    top_terms=10,
                )