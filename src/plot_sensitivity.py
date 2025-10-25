# src/plot_sensitivity.py

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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def plot_sensitivity(
    features_df: pd.DataFrame,
    outdir: str = "results/Benchmarking BioID DIA",
    initial_positives_list=(1, 5, 10, 15, 20, 25, 30),
    initial_negatives_list=(100, 200, 500, 1000),
    feature_columns=(
        "log_fold_change",
        "snr",
        "mean_diff",
        "median_diff",
        "replicate_fold_change_sd",
        "bait_cv",
        "bait_control_sd_ratio",
        "zero_or_neg_fc",
    ),
    bait_col: str = "Bait",
    prey_col: str = "Prey",
    composite_col: str = "composite_score",
    single_rep_flag_col: str = "single_rep_flag",
    prob_col: str = "predicted_probability",
    fdr_col: str = "FDR",
    fdr_threshold_for_hits: float = 0.01,
    hist_bins_main: int = 100,
    hist_range=(0, 1),
    seed: int = 42,
    save_prefix: str = "",
):
    """
    Sensitivity analysis over (initial_positives, initial_negatives) for whisper PU labeling.

    For each (N_pos, N_neg):
      - Cluster baits (top-50 composite std), pick "strong" cluster
      - Label top N_pos per strong bait as positives (excl. single-replicate spikes if available)
      - Label N_neg from bottom per bait as negatives
      - Train scaler + Bagging(RandomForest) on labeled data
      - Score all, build decoy by per-bait feature shuffling
      - Estimate monotone FDR from decoys vs reals
      - Compute robustness metrics:
          * hits below FDR≤fdr_threshold_for_hits
          * KL divergence between real & decoy score histograms
          * 1D Wasserstein distance
          * ROC AUC (real vs decoy score discrimination)
          * median score difference (real - decoy)

    Saves a results table and a multi-panel line figure.

    Returns
    -------
    dict with paths to CSV and PDF.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # basic checks
    needed = set([bait_col, prey_col, composite_col]) | set(feature_columns)
    missing = [c for c in needed if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df missing columns: {missing}")

    df_base = (
        features_df.copy()
        .sort_values([bait_col, prey_col])
        .reset_index(drop=True)
    )

    X_all = df_base[list(feature_columns)].to_numpy()

    results_rows = []
    for N_pos in initial_positives_list:
        for N_neg in initial_negatives_list:
            print(f"[whisper sensitivity] Training with {N_pos} positives and {N_neg} negatives")

            df = df_base.copy()

            # ---------- Bait clustering (top-50 composite std) ----------
            top50_std = (
                df.groupby(bait_col)[composite_col]
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

            bait_cluster_map = {b: int(c) for b, c in zip(bait_names, clusters)}

            # choose "strong" cluster: larger size; tie-break by higher mean std
            uniq = np.unique(clusters)
            size_by_c = {c: int(np.sum(clusters == c)) for c in uniq}
            mean_by_c = {c: float(bait_scores[clusters == c].mean()) for c in uniq}
            max_size = max(size_by_c.values())
            cands = [c for c, n in size_by_c.items() if n == max_size]
            strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: mean_by_c[c])

            strong_baits = [b for b in bait_names if bait_cluster_map[b] == strong_cluster_id]
            bait_pos_quota = {b: (N_pos if b in strong_baits else 0) for b in df[bait_col].unique()}

            # ---------- Label positives / negatives ----------
            y_labels = pd.Series(0, index=df.index)
            has_single_flag = single_rep_flag_col in df.columns

            for bait in df[bait_col].unique():
                bait_df = df[df[bait_col] == bait]
                n_pos = bait_pos_quota[bait]
                if n_pos > 0:
                    ranked = bait_df.sort_values(composite_col, ascending=False)
                    if has_single_flag:
                        ranked = ranked[ranked[single_rep_flag_col] != 1]
                    top_pos_idx = ranked.index[:n_pos]
                    y_labels.loc[top_pos_idx] = 1

                # negatives from bottom (exclude chosen positives)
                remaining = bait_df.drop(index=y_labels[y_labels == 1].index, errors="ignore")
                bottom_neg_idx = remaining[composite_col].nsmallest(N_neg).index
                y_labels.loc[bottom_neg_idx] = -1

            labeled_idx = y_labels[y_labels != 0].index
            X_train = X_all[labeled_idx]
            y_train = y_labels.loc[labeled_idx].to_numpy()

            # ---------- Train ----------
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            model = BaggingClassifier(estimator=rf, n_estimators=100, random_state=seed)
            model.fit(X_train_scaled, y_train)

            X_scaled = scaler.transform(X_all)
            df[prob_col] = model.predict_proba(X_scaled)[:, 1]

            # ---------- Decoys per bait: feature shuffling ----------
            decoy_probs = []
            for j, bait in enumerate(df[bait_col].unique()):
                # fixed, bait-specific seed for reproducibility
                rng_bait = np.random.default_rng(seed + j)
                bait_df = df[df[bait_col] == bait].copy()
                df_decoy = bait_df.copy()
                for col in feature_columns:
                    vals = df_decoy[col].to_numpy()
                    rng_bait.shuffle(vals)
                    df_decoy[col] = vals
                X_decoy = scaler.transform(df_decoy[list(feature_columns)].to_numpy())
                decoy_probs.append(model.predict_proba(X_decoy)[:, 1])
            decoy_probs = np.concatenate(decoy_probs) if len(decoy_probs) else np.array([])

            real_probs = df[prob_col].to_numpy()

            # ---------- Monotone FDR from decoys ----------
            unique_probs = np.unique(real_probs)
            raw_fdr = {}
            for p in unique_probs:
                real_above = np.sum(real_probs >= p)
                decoy_above = np.sum(decoy_probs >= p) if decoy_probs.size else 0
                ratio = (decoy_above / real_above) if real_above > 0 else 1.0
                raw_fdr[p] = min(ratio, 1.0)

            sorted_probs = np.sort(unique_probs)
            mono_fdr = {sorted_probs[0]: raw_fdr[sorted_probs[0]]}
            for i in range(1, len(sorted_probs)):
                p = sorted_probs[i]
                prev_p = sorted_probs[i - 1]
                mono_fdr[p] = min(raw_fdr[p], mono_fdr[prev_p])

            df[fdr_col] = df[prob_col].map(mono_fdr)

            # ---------- Metrics ----------
            try:
                auc_score = roc_auc_score(
                    np.concatenate([np.ones_like(real_probs), np.zeros_like(decoy_probs)]),
                    np.concatenate([real_probs, decoy_probs]),
                ) if decoy_probs.size else np.nan
            except Exception:
                auc_score = np.nan

            fdr_hits = int((df[fdr_col] <= fdr_threshold_for_hits).sum())
            w_dist = wasserstein_distance(real_probs, decoy_probs) if decoy_probs.size else np.nan

            # KL on discretized histograms
            hist_real, _ = np.histogram(real_probs, bins=hist_bins_main, range=hist_range, density=True)
            hist_decoy, _ = np.histogram(decoy_probs, bins=hist_bins_main, range=hist_range, density=True) if decoy_probs.size else (np.zeros_like(hist_real), None)
            hist_real = hist_real + 1e-10
            hist_decoy = hist_decoy + 1e-10
            kl_div = float(entropy(hist_real, hist_decoy))

            median_diff = float(np.nanmedian(real_probs) - (np.nanmedian(decoy_probs) if decoy_probs.size else np.nan))
            # mean #hits per bait at chosen FDR
            bait_hit_counts = (
                df[df[fdr_col] <= fdr_threshold_for_hits][bait_col].value_counts().mean()
                if (df[fdr_col] <= fdr_threshold_for_hits).any()
                else 0.0
            )

            results_rows.append(
                {
                    "initial_positives": int(N_pos),
                    "initial_negatives": int(N_neg),
                    "fdr_hits": fdr_hits,
                    "kl_divergence": float(kl_div),
                    "wasserstein_distance": float(w_dist) if w_dist == w_dist else np.nan,
                    "roc_auc": float(auc_score) if auc_score == auc_score else np.nan,
                    "median_score_difference": float(median_diff) if median_diff == median_diff else np.nan,
                    "mean_hits_per_bait_at_1pctFDR": float(bait_hit_counts),
                }
            )

    # ---------- Save table ----------
    res_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(outdir, f"pu_learning_optimization{save_prefix}.csv")
    res_df.to_csv(csv_path, index=False)

    # ---------- Plot ----------
    df_plot = res_df.copy()

    metrics = [
        ("fdr_hits", "Hits Below 1% FDR"),
        ("kl_divergence", "KL Divergence (Real vs Decoy)"),
        ("wasserstein_distance", "Wasserstein Distance"),
        ("roc_auc", "ROC AUC (Real vs Decoy)"),
        ("median_score_difference", "Median Score Difference (Real - Decoy)"),
        ("mean_hits_per_bait_at_1pctFDR", "Mean #Hits/Bait @1% FDR"),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for ax, (metric, title) in zip(axes, metrics):
        sns.lineplot(
            data=df_plot,
            x="initial_positives",
            y=metric,
            hue="initial_negatives",
            marker="o",
            ax=ax,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Initial Positives")
        ax.set_ylabel(title.split("(")[0].strip())

    plt.tight_layout()
    pdf_path = os.path.join(outdir, f"pu_learning_optimization_plot{save_prefix}.pdf")
    plt.savefig(pdf_path, dpi=300)
    plt.close()

    return {"csv": csv_path, "pdf": pdf_path}
