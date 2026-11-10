"""Peptide- and fragment-level WHISPER against SAINTq and the heuristic score
(Fig. 2d-e)."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]

# Set by run().
LEVEL = None
SCORES_FILE = None
SAINTQ_FILE = None
OUTDIR = None
MATRIX_FILE, MATRIX_SEP, MATRIX_PROTEIN_COL = None, None, None
GROUND_TRUTHS = []

STEP_SIZE = 10
N_BOOTSTRAP = 100
DEPTHS = {"Total AUC": None, "AUC@Top300": 300}
DEPTH_LABELS = list(DEPTHS)

METHOD_COLORS = {
    "WHISPER": "#003f5c",
    "SAINTq": "#ffb347",
    "Composite score": "#7a5195",
    "Labeling score": "#7a5195",
}


def add_missing_preys(df, features_df, prey_col):

    updated = pd.DataFrame()
    for bait in features_df["Bait"].unique():
        want = set(features_df[features_df["Bait"] == bait]["Prey"])
        have = set(df[df["Bait"] == bait][prey_col])
        missing = want - have
        if missing:
            print(f"Adding {len(missing)} missing preys for bait {bait}.")
            updated = pd.concat([updated, pd.DataFrame({
                "Bait": [bait] * len(missing),
                prey_col: list(missing),
                "BFDR": [1.0] * len(missing),
                "AvgP": [0.0] * len(missing)})], ignore_index=True)
        updated = pd.concat([updated, df[df["Bait"] == bait]],
                            ignore_index=True)
    return updated


def tie_averaged_labels(score, true):

    score = np.asarray(score, dtype=float)
    score = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-score, kind="stable")
    s, t = score[order], np.asarray(true, dtype=float)[order]
    if len(s) == 0:
        return t
    block = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    return (np.bincount(block, weights=t) / np.bincount(block))[block]


def _pr_auc_from_sorted(true_sorted, step_size=STEP_SIZE):

    n = len(true_sorted)
    if n == 0:
        return 0.0
    cum = np.cumsum(true_sorted)
    total_pos = cum[-1]
    if total_pos <= 0:
        return 0.0
    ks = np.arange(step_size, n + step_size, step_size)
    tp = cum[np.minimum(ks, n) - 1]
    return auc(tp / total_pos, tp / ks)


def paired_bootstrap_aucs(specs, depths, n_bootstrap=N_BOOTSTRAP, seed=0):

    keys = [list(zip(df["Bait"].astype(str), df[prey_col].astype(str)))
            for _n, df, prey_col, _sc in specs]

    common = set(keys[0])
    for k in keys[1:]:
        common &= set(k)
    seen, order = set(), []
    for k in keys[0]:
        if k in common and k not in seen:
            seen.add(k)
            order.append(k)

    aligned = {}
    for (name, df, prey_col, score_col), k in zip(specs, keys):
        d = df.copy()
        d["_pair_key"] = k
        d = d.drop_duplicates("_pair_key").set_index("_pair_key").reindex(order)
        score = pd.to_numeric(d[score_col], errors="coerce").to_numpy(dtype=float)
        # An unscored pair ranks last rather than dropping out, so every method
        # is measured on an identical pair set.
        score = np.where(np.isnan(score), -np.inf, score)
        aligned[name] = (score, d["true_label"].to_numpy(dtype=float))

    rng = np.random.default_rng(seed)
    n = len(order)
    out = {name: {lab: np.empty(n_bootstrap) for lab in depths}
           for name in aligned}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)         
        for name, (score, true) in aligned.items():
            t_sorted = tie_averaged_labels(score[idx], true[idx])
            for lab, topn in depths.items():
                out[name][lab][b] = _pr_auc_from_sorted(
                    t_sorted if topn is None else t_sorted[:topn])
    return out


def paired_difference_ci(boot, a, b, label, alpha=0.05):
    d = boot[a][label] - boot[b][label]
    return {"mean_difference": float(d.mean()),
            "ci_low": float(np.percentile(d, 100 * alpha / 2)),
            "ci_high": float(np.percentile(d, 100 * (1 - alpha / 2))),
            "frac_replicates_positive": float((d > 0).mean())}


def pr_curve(df, score_col, true_col, step=STEP_SIZE):
    # Ties take their block's mean hit rate, as in the bootstrap AUCs.
    t = tie_averaged_labels(
        pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float),
        df[true_col].to_numpy(dtype=float))
    cum = np.cumsum(t)
    total_pos = cum[-1] if len(cum) else 0
    if total_pos <= 0:
        return np.array([0.0]), np.array([0.0])
    ks = np.arange(step, len(t) + step, step)
    tp = cum[np.minimum(ks, len(t)) - 1]
    return tp / total_pos, tp / ks


def is_any_gene_in_gold(prey_str, gold_set):
    if pd.isna(prey_str):
        return 0
    return int(any(gene.strip() in gold_set for gene in str(prey_str).split(";")))


def detected_proteins():

    p = pd.read_csv(MATRIX_FILE, sep=MATRIX_SEP,
                    usecols=[MATRIX_PROTEIN_COL])[MATRIX_PROTEIN_COL]
    p = p[p.notna()].astype(str).str.strip()
    return set(p[~p.isin(["", "0", "nan", "None"])])


def load_methods():
    scores = pd.read_csv(SCORES_FILE)
    prot = (scores
            .groupby(["Bait", "Protein"], as_index=False)
            .agg(predicted_probability=("predicted_probability", "max"),
                 composite_score=("composite_score", "max"),
                 FDR=("FDR", "min"))
            .rename(columns={"Protein": "Prey"}))
    prot["composite_score_norm"] = MinMaxScaler().fit_transform(
        prot[["composite_score"]]).flatten()

    saintq = pd.read_csv(SAINTQ_FILE, sep="\t")
    # SAINTq writes the prey column as Prey in current builds and PreyGene in
    # older ones; both are accepted so old result files still load.
    saintq = saintq.rename(columns={"PreyGene": "Prey"})
    saintq = add_missing_preys(saintq, prot, "Prey")
    # Every method is scored on WHISPER's pairs, curves and bars alike. SAINTq
    # also scores proteins the training's minimum-evidence filter removed;
    # those pairs are left out here so the curve and the bootstrap see the
    # same pair set.
    keep = set(zip(prot["Bait"].astype(str), prot["Prey"].astype(str)))
    saintq = saintq[[k in keep for k in zip(saintq["Bait"].astype(str),
                                            saintq["Prey"].astype(str))]
                    ].reset_index(drop=True)

    print(f"{len(prot)} bait-protein pairs from {len(scores)} {LEVEL} rows; "
          f"SAINTq covers {len(saintq)} after padding")
    frames = {
        "WHISPER": (prot, "Prey", "predicted_probability"),
        "SAINTq": (saintq, "Prey", "AvgP"),
        "Composite score": (prot, "Prey", "composite_score_norm"),
    }
    return prot, frames


def bar_figure(auc_table, methods, out_pdf):
    melted = auc_table.melt(id_vars="Method", value_vars=DEPTH_LABELS,
                            var_name="AUC Type", value_name="AUC")
    melted["Std"] = auc_table.melt(
        id_vars="Method", value_vars=[f"{d} Std" for d in DEPTH_LABELS],
        var_name="AUC Type", value_name="Std")["Std"]

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=melted, x="AUC Type", y="AUC", hue="Method",
                     palette=METHOD_COLORS, order=DEPTH_LABELS,
                     hue_order=methods)
    for i, depth in enumerate(DEPTH_LABELS):
        sub = (melted[melted["AUC Type"] == depth].set_index("Method")
               .reindex(methods).reset_index())
        xs = [i - 0.4 + (j + 0.5) * (0.8 / len(sub)) for j in range(len(sub))]
        ax.errorbar(xs, sub["AUC"], yerr=sub["Std"], fmt="none", c="black",
                    capsize=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC Value", fontsize=26)
    ax.set_xlabel("", fontsize=26)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)

    ax.set_title(f"Benchmarking BioID DIA ({LEVEL} level)", fontsize=18)
    plt.legend(title="Method", ncol=len(methods), loc="lower center",
               bbox_to_anchor=(0.5, -0.3), fontsize=14)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def curve_figure(curves, methods, aucs, stds, out_pdf):
    plt.figure(figsize=(8, 8))
    for m in methods:
        rec, prec = curves[m]
        plt.plot(rec, prec, marker="o", markersize=2,
                 color=METHOD_COLORS[m],
                 label=f"{m} (AUC: {aucs[m]:.4f} ± {stds[m]:.4f})")
    plt.xlabel("Recall", fontsize=26)
    plt.ylabel("Precision", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(0, 1)
    plt.title("Overall Precision vs. Recall (All Baits Combined)", fontsize=20)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    prot, frames = load_methods()
    methods = list(frames)
    identified = detected_proteins()
    print(f"{len(identified)} proteins detected in {MATRIX_FILE}")

    for gt_source, gt_file in GROUND_TRUTHS:
        gt = pd.read_csv(gt_file)
        gt = gt[gt["Prey"].isin(identified)]
        baits = list(prot["Bait"].unique())
        gold = {b: set(gt[gt["Bait"] == b]["Prey"]) for b in baits}

        specs, curves = [], {}
        for m in methods:
            df, prey_col, score_col = frames[m]
            d = df.copy()
            d["true_label"] = [is_any_gene_in_gold(p, gold.get(b, set()))
                               for b, p in zip(d["Bait"], d[prey_col])]
            specs.append((m, d, prey_col, score_col))
            curves[m] = pr_curve(d, score_col, "true_label")

        n_pos, n_pairs = int(specs[0][1]["true_label"].sum()), len(specs[0][1])
        print(f"\n{gt_source}: {n_pos}/{n_pairs} pairs annotated")

        boot = paired_bootstrap_aucs(specs, DEPTHS, n_bootstrap=N_BOOTSTRAP,
                                     seed=0)
        rows = []
        for m in methods:
            row = {"Method": m}
            for depth in DEPTH_LABELS:
                a = boot[m][depth]
                row[depth] = float(a.mean())
                row[f"{depth} Std"] = float(a.std())
            rows.append(row)
        auc_table = pd.DataFrame(rows)
        auc_table["ground_truth"] = gt_source
        auc_table.to_csv(
            f"{OUTDIR}/auc_source_data_{gt_source}_main_methods_{LEVEL}.csv",
            index=False)

        diff_rows = []
        for depth in DEPTH_LABELS:
            for other in [m for m in methods if m != "WHISPER"]:
                r = paired_difference_ci(boot, "WHISPER", other, depth)
                r.update({"comparison": f"WHISPER - {other}",
                          "reference_method": other, "depth": depth,
                          "ground_truth": gt_source})
                diff_rows.append(r)
        pd.DataFrame(diff_rows).to_csv(
            f"{OUTDIR}/auc_paired_differences_{gt_source}_{LEVEL}.csv",
            index=False)

        aucs = {m: auc_table.set_index("Method").loc[m, "Total AUC"]
                for m in methods}
        stds = {m: auc_table.set_index("Method").loc[m, "Total AUC Std"]
                for m in methods}
        for m in methods:
            print(f"  {m:<16} Total AUC = {aucs[m]:.4f} ± {stds[m]:.4f} | "
                  f"AUC@300 = "
                  f"{auc_table.set_index('Method').loc[m, 'AUC@Top300']:.4f}")

        main_methods = ["WHISPER", "SAINTq"]
        curve_figure(curves, main_methods, aucs, stds,
                     f"{OUTDIR}/overall_pr_curve_combined_{gt_source}"
                     f"_main_methods_{LEVEL}.pdf")
        bar_figure(auc_table[auc_table["Method"].isin(main_methods)],
                   main_methods,
                   f"{OUTDIR}/auc_barplot_combined_{gt_source}"
                   f"_main_methods_{LEVEL}.pdf")

        vs_comp = ["WHISPER", "Composite score"]
        curve_figure(curves, vs_comp, aucs, stds,
                     f"{OUTDIR}/overall_pr_curve_combined_{gt_source}"
                     f"_whisper_vs_labeling_{LEVEL}.pdf")
        bar_figure(auc_table[auc_table["Method"].isin(vs_comp)], vs_comp,
                   f"{OUTDIR}/auc_barplot_combined_{gt_source}"
                   f"_whisper_vs_labeling_{LEVEL}.pdf")

    print(f"\nDone. Figures and source data written to {OUTDIR}")


def run(level, scores_file, saintq_file, matrix_file, references, outdir):
    global LEVEL, SCORES_FILE, SAINTQ_FILE, OUTDIR, MATRIX_FILE, MATRIX_SEP
    global MATRIX_PROTEIN_COL, GROUND_TRUTHS
    LEVEL, SCORES_FILE, SAINTQ_FILE, OUTDIR = level, scores_file, saintq_file, outdir
    MATRIX_FILE = matrix_file
    MATRIX_SEP = "\t" if matrix_file.endswith((".tsv", ".txt")) else ","
    MATRIX_PROTEIN_COL = "protein" if level == "peptide" else "Protein"
    GROUND_TRUTHS = list(references)
    main()
