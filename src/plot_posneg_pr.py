# src/plot_posneg_pr.py

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

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")


def plot_posneg_pr(
    features_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    outdir: str = "results/Benchmarking BioID DIA",
    gt_source: str = "go",
    positives_list=(5, 10, 15, 20, 25, 30),
    negatives_list=(100, 200, 500, 1000),
    feature_columns=(
        "log_fold_change", "snr", "mean_diff", "median_diff",
        "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio", "zero_or_neg_fc",
    ),
    bait_col: str = "Bait",
    prey_col: str = "Prey",
    heuristic_col: str = "heuristic_score",
    single_rep_flag_col: str = "single_rep_flag",
    n_bootstrap: int = 100,
    topn_for_auc: int = 300,
    seed: int = 42,
    cache_csv: bool = True,
):
    """
    Sweep over (Positives, Negatives) labeling choices; train whisper (PU) each time;
    compute PR AUC and AUC@TopN vs GO/BioGRID ground truth; save CSV + bar plots.

    Returns
    -------
    dict: {"csv": <path>, "pdf_auc": <path>, "pdf_auc_topn": <path>}
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _run_pu(df_real: pd.DataFrame, N_pos=15, N_neg=200):
        np.random.seed(seed)

        # cluster baits by top-50 std of heuristic
        top50_std = (
            df_real.groupby(bait_col)[heuristic_col]
            .apply(lambda s: s.nlargest(50).std())
            .fillna(0.0)
        )
        bait_names = top50_std.index.to_numpy()
        bait_scores = top50_std.to_numpy().reshape(-1, 1)

        if bait_names.size > 2:
            Z = linkage(bait_scores, method="ward")
            clusters = fcluster(Z, t=2, criterion="maxclust")
        else:
            clusters = np.ones(bait_names.size, dtype=int)

        # choose strong cluster: larger size; tie-break by higher mean std
        uniq = np.unique(clusters)
        size_by_c = {c: int(np.sum(clusters == c)) for c in uniq}
        mean_by_c = {c: float(bait_scores[clusters == c].mean()) for c in uniq}
        max_size = max(size_by_c.values())
        cands = [c for c, n in size_by_c.items() if n == max_size]
        strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: mean_by_c[c])

        strong_baits = [b for b, c in zip(bait_names, clusters) if c == strong_cluster_id]
        pos_quota = {b: (N_pos if b in strong_baits else 0) for b in df_real[bait_col].unique()}

        # label
        y_labels = pd.Series(0, index=df_real.index)
        has_single_flag = single_rep_flag_col in df_real.columns

        for b in df_real[bait_col].unique():
            sub = df_real[df_real[bait_col] == b]
            n_pos = pos_quota[b]
            if n_pos > 0:
                ranked = sub.sort_values(heuristic_col, ascending=False)
                if has_single_flag:
                    ranked = ranked[ranked[single_rep_flag_col] != 1]
                top_idx = ranked.index[:n_pos]
                y_labels.loc[top_idx] = 1

            # negatives from bottom excluding chosen positives
            remaining = sub.drop(index=y_labels[y_labels == 1].index, errors="ignore")
            neg_idx = remaining[heuristic_col].nsmallest(N_neg).index
            y_labels.loc[neg_idx] = -1

        labeled_idx = y_labels[y_labels != 0].index
        X_train = df_real.loc[labeled_idx, list(feature_columns)].to_numpy()
        y_train = y_labels.loc[labeled_idx].to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        bagged = BaggingClassifier(estimator=rf, n_estimators=100, random_state=seed)
        bagged.fit(X_train_scaled, y_train)

        X_all = df_real[list(feature_columns)].to_numpy()
        X_scaled = scaler.transform(X_all)
        probs = bagged.predict_proba(X_scaled)[:, 1]
        return probs

    def _calc_pr_topn(df, score_col, true_col, step_size=10):
        dfs = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        n_pos_total = dfs[true_col].sum()
        if n_pos_total == 0:
            return np.array([]), np.array([])
        precision, recall = [], []
        topn_list = np.arange(step_size, len(dfs) + step_size, step_size)
        for topn in topn_list:
            sel = dfs.iloc[:min(topn, len(dfs))]
            tp = int(sel[true_col].sum())
            fp = len(sel) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / n_pos_total
            precision.append(prec)
            recall.append(rec)
        return np.asarray(recall), np.asarray(precision)

    def _bootstrap_auc(recall, precision, n_boot=100):
        if recall.size == 0:
            return np.nan, np.nan
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, recall.size, recall.size)
            r = recall[idx]; p = precision[idx]
            order = np.argsort(r)
            vals.append(auc(r[order], p[order]))
        return float(np.mean(vals)), float(np.std(vals))

    def _bootstrap_auc_at_topn(df, score_col, true_col, topn=300, n_boot=100):
        if len(df) == 0:
            return np.nan, np.nan
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(df), len(df))
            dfs = df.iloc[idx].sort_values(score_col, ascending=False).reset_index(drop=True)
            dft = dfs.iloc[:min(topn, len(dfs))]
            r, p = _calc_pr_topn(dft, score_col, true_col, step_size=10)
            if r.size > 1:
                order = np.argsort(r)
                vals.append(auc(r[order], p[order]))
        return float(np.mean(vals)), float(np.std(vals))

    # -----------------------------
    # Ground truth mapping
    # -----------------------------
    gt_df = gt_df[[bait_col, prey_col]].dropna().copy()
    gt_df.columns = [bait_col, prey_col]
    gt_dict = {b: set(gp[prey_col]) for b, gp in gt_df.groupby(bait_col)}

    # restrict to identified preys
    identified = set(features_df[prey_col].unique())
    gt_dict = {b: (s & identified) for b, s in gt_dict.items()}

    # -----------------------------
    # Cache path
    # -----------------------------
    csv_name = f"whisper_pr_by_posneg_{gt_source}.csv"
    csv_path = os.path.join(outdir, csv_name)

    if cache_csv and os.path.exists(csv_path):
        auc_df_full = pd.read_csv(csv_path)
        # keep consistent column names if older cache versions exist
        rename_map = {
            "npos": "Positives",
            "nneg": "Negatives",
            "mean_auc": "Mean AUC",
            "std_auc": "Std AUC",
            "mean_auc_top300": "Mean AUC@Top300",
            "std_auc_top300": "Std AUC@Top300",
        }
        auc_df_full = auc_df_full.rename(columns={k: v for k, v in rename_map.items() if k in auc_df_full.columns})
    else:
        # compute fresh
        results = []
        for npos in positives_list:
            for nneg in negatives_list:
                print(f"[whisper pos/neg] running PU with {npos} positives, {nneg} negatives")
                df_run = features_df.copy()
                df_run["predicted_probability"] = _run_pu(df_run, N_pos=npos, N_neg=nneg)

                # truth labels
                df_run["true_label"] = df_run.apply(
                    lambda r: 1 if r[prey_col] in gt_dict.get(r[bait_col], set()) else 0,
                    axis=1,
                )

                r, p = _calc_pr_topn(df_run, "predicted_probability", "true_label")
                mean_auc, std_auc = _bootstrap_auc(r, p, n_boot=n_bootstrap)
                mean_auc_top300, std_auc_top300 = _bootstrap_auc_at_topn(
                    df_run, "predicted_probability", "true_label", topn=topn_for_auc, n_boot=n_bootstrap
                )

                results.append(
                    {
                        "Positives": int(npos),
                        "Negatives": int(nneg),
                        "Mean AUC": float(mean_auc),
                        "Std AUC": float(std_auc),
                        "Mean AUC@Top300": float(mean_auc_top300),
                        "Std AUC@Top300": float(std_auc_top300),
                    }
                )

        auc_df_full = pd.DataFrame(results)
        auc_df_full.to_csv(csv_path, index=False)

    # split views for plotting
    auc_df      = auc_df_full[["Positives", "Negatives", "Mean AUC", "Std AUC"]].copy()
    auc_top300  = auc_df_full[["Positives", "Negatives", "Mean AUC@Top300", "Std AUC@Top300"]].copy()

    # -----------------------------
    # Plot: AUC vs (Positives, Negatives)
    # -----------------------------
    pdf_auc = os.path.join(outdir, f"whisper_auc_by_posneg_{gt_source}.pdf")
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=auc_df, x="Positives", y="Mean AUC", hue="Negatives", palette="Blues", errorbar=None)
    # add error bars by hand
    for bar, (_, row) in zip(ax.patches, auc_df.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.errorbar(x, y, yerr=row["Std AUC"], fmt="none", ecolor="black", capsize=3)
    plt.title("whisper (PU): AUC across Pos/Neg labeling", fontsize=18)
    plt.xlabel("Number of Positives", fontsize=16)
    plt.ylabel("AUC", fontsize=16)
    plt.legend(title="Negatives", fontsize=11, title_fontsize=12, loc=(1.02, 1))
    plt.tight_layout()
    plt.savefig(pdf_auc, dpi=300)
    plt.close()

    # -----------------------------
    # Plot: AUC@TopN vs (Positives, Negatives)
    # -----------------------------
    pdf_auc_topn = os.path.join(outdir, f"whisper_auc_top{topn_for_auc}_by_posneg_{gt_source}.pdf")
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=auc_top300, x="Positives", y="Mean AUC@Top300", hue="Negatives", palette="Blues", errorbar=None)
    for bar, (_, row) in zip(ax.patches, auc_top300.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.errorbar(x, y, yerr=row["Std AUC@Top300"], fmt="none", ecolor="black", capsize=3)
    plt.title(f"whisper (PU): AUC@Top{topn_for_auc} across Pos/Neg labeling", fontsize=18)
    plt.xlabel("Number of Positives", fontsize=16)
    plt.ylabel(f"AUC@Top{topn_for_auc}", fontsize=16)
    plt.legend(title="Negatives", fontsize=11, title_fontsize=12, loc=(1.02, 1))
    plt.tight_layout()
    plt.savefig(pdf_auc_topn, dpi=300)
    plt.close()

    return {"csv": csv_path, "pdf_auc": pdf_auc, "pdf_auc_topn": pdf_auc_topn}
