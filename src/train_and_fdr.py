import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
import os

warnings.filterwarnings("ignore")


def run_training_and_fdr(    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
) -> pd.DataFrame:
    """
    Trains a learning model using Bagging with Random Forest, and estimates FDR using bait-specific decoys.

    Parameters:
        df_real (pd.DataFrame): Feature-engineered dataframe with 'composite_score', 'Bait', and other features.
        initial_positives (int): Number of initial positives per strong bait (default = 10).
        initial_negatives (int): Number of negatives per bait (default = 200).

    Returns:
        df_real (pd.DataFrame): The same dataframe with 'predicted_probability', 'FDR', and 'global_cv_flag' columns.
        all_decoy_probs (np.ndarray): Array of predicted probabilities from decoy distributions (for plotting).
    """

    df_real = features_df.copy().sort_values(["Bait", "Prey"]).reset_index(drop=True)
    np.random.seed(42)

    feature_columns = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff',
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
        'zero_or_neg_fc',
    ]
    X_real = df_real[feature_columns].values

    # === Hierarchical clustering of baits ===
    bait_top50_stds = {
        bait: df_real[df_real['Bait'] == bait]['composite_score'].nlargest(50).std()
        for bait in df_real['Bait'].unique()
    }
    bait_names  = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values())).reshape(-1, 1)

    if len(bait_names) > 2:
        linkage_matrix = linkage(bait_scores, method='ward')
        clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    else:
        clusters = np.ones(len(bait_names), dtype=int)

    bait_cluster_map = {b: int(c) for b, c in zip(bait_names, clusters)}

    # determine strong cluster (largest size, then highest mean std)
    unique_clusters = np.unique(clusters)
    cluster_sizes = {c: int(np.sum(clusters == c)) for c in unique_clusters}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in unique_clusters}
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]
    strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cluster_means[c])

    strong_baits = [b for b in bait_names if bait_cluster_map[b] == strong_cluster_id]

    # === Assign positives ===
    bait_scaled_positives = {
        bait: (initial_positives if bait in strong_baits else 0)
        for bait in df_real['Bait'].unique()
    }

    y_labels = pd.Series(0, index=df_real.index)
    for bait in df_real['Bait'].unique():
        bait_df = df_real[df_real['Bait'] == bait]
        N_pos = bait_scaled_positives[bait]

        if N_pos > 0:
            ranked = bait_df.sort_values('composite_score', ascending=False)
            elig_pos = ranked[ranked['single_rep_flag'] != 1]
            top_pos = elig_pos.index[:N_pos]
            y_labels.loc[top_pos] = 1

            remaining = bait_df.drop(index=top_pos, errors='ignore')
            bottom_neg = remaining['composite_score'].nsmallest(initial_negatives).index
            y_labels.loc[bottom_neg] = -1

    # === Train classifier ===
    labeled_idx = y_labels[y_labels != 0].index
    X_train = X_real[labeled_idx]
    y_train = y_labels.loc[labeled_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    bag_rf = BaggingClassifier(estimator=rf, n_estimators=100, random_state=42)
    bag_rf.fit(X_train_scaled, y_train)

    X_scaled = scaler.transform(X_real)
    df_real["predicted_probability"] = bag_rf.predict_proba(X_scaled)[:, 1]

    # === Decoy-based FDR estimation ===
    all_decoy_probs = []
    for i, bait in enumerate(df_real['Bait'].unique()):
        np.random.seed(42 + i)
        bait_df = df_real[df_real['Bait'] == bait]
        df_decoy = bait_df.copy()
        for col in feature_columns:
            df_decoy[col] = np.random.permutation(df_decoy[col].values)
        X_decoy = scaler.transform(df_decoy[feature_columns].values)
        all_decoy_probs.extend(bag_rf.predict_proba(X_decoy)[:, 1])

    all_decoy_probs = np.array(all_decoy_probs)
    unique_probs = np.unique(df_real["predicted_probability"].values)

    raw_fdr = {}
    for p in unique_probs:
        n_real = (df_real["predicted_probability"] >= p).sum()
        n_decoy = (all_decoy_probs >= p).sum()
        raw_fdr[p] = min(n_decoy / n_real if n_real > 0 else 1.0, 1.0)

    sorted_probs = np.sort(unique_probs)
    fdr_map = {sorted_probs[0]: raw_fdr[sorted_probs[0]]}
    for i in range(1, len(sorted_probs)):
        curr = sorted_probs[i]
        prev = sorted_probs[i - 1]
        fdr_map[curr] = min(raw_fdr[curr], fdr_map[prev])

    df_real["FDR"] = df_real["predicted_probability"].map(fdr_map)

    # === Global CV background flag ===
    if "global_cv" in df_real.columns:
        cv_thresh = np.nanpercentile(df_real["global_cv"], 25)
        df_real["global_cv_flag"] = df_real["global_cv"].apply(
            lambda cv: "likely background" if cv <= cv_thresh else ""
        )
    else:
        df_real["global_cv_flag"] = ""

    # === Save ===
    df_real.to_csv("whisper_protein_scores.csv", index=False)
    return df_real










def run_train_and_fdr_peptide(
    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
    random_state: int = 42,
    save_dir: str = ".",
    peptide_out: str = "whisper_peptide_scores.csv",
    protein_out: str = "whisper_protein_scores_from_peptides.csv",
    aggregate_strategy: str = "max",   # "max" or "mean" for peptide->protein prob aggregation
):
    """
    Train a model on PEPTIDE-level features, compute bait-specific decoy FDR,
    and aggregate to PROTEIN-level scores per bait.

    Expected columns in `features_df`:
      - Bait, Protein, Peptide
      - composite_score, global_cv (optional), single_rep_flag (optional)
      - Feature columns:
          ['log_fold_change','snr','mean_diff','median_diff',
           'replicate_fold_change_sd','bait_cv','bait_control_sd_ratio','zero_or_neg_fc']

    Saves:
      - <save_dir>/<peptide_out>: peptide-level scores with FDR
      - <save_dir>/<protein_out>: protein-level aggregation per bait

    Returns:
      (peptide_df, protein_df)
    """
    rng = np.random.RandomState(random_state)

    # ----- Stable sort -----
    df = features_df.copy().sort_values(["Bait", "Protein", "Peptide"]).reset_index(drop=True)

    feature_columns = [
        "log_fold_change", "snr", "mean_diff", "median_diff",
        "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
        "zero_or_neg_fc",
    ]
    X = df[feature_columns].values

    # ----- Cluster baits to identify "strong" cluster (same logic as protein) -----
    bait_top50_stds = {
        b: df[df["Bait"] == b]["composite_score"].nlargest(50).std()
        for b in df["Bait"].unique()
    }
    bait_names = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values()), dtype=float).reshape(-1, 1)

    if len(bait_names) > 2:
        Z = linkage(bait_scores, method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")
    else:
        clusters = np.ones(len(bait_names), dtype=int)

    cluster_sizes = {c: int(np.sum(clusters == c)) for c in np.unique(clusters)}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in np.unique(clusters)}
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]
    strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cluster_means[c])
    strong_baits = [b for b, c in zip(bait_names, clusters) if c == strong_cluster_id]

    # ----- Pseudo-labels -----
    y = pd.Series(0, index=df.index)  # 0=unlabeled, 1=pos, -1=neg
    bait_pos_quota = {b: (initial_positives if b in strong_baits else 0) for b in df["Bait"].unique()}

    for bait in df["Bait"].unique():
        sub = df[df["Bait"] == bait].copy()
        n_pos = bait_pos_quota[bait]
        if n_pos > 0:
            ranked = sub.sort_values("composite_score", ascending=False)
            # exclude single-replicate spikes if column exists
            elig = ranked[ranked.get("single_rep_flag", 0) != 1]
            pos_idx = elig.index[:n_pos]
            y.loc[pos_idx] = 1

            remaining = sub.drop(index=pos_idx, errors="ignore")
            neg_idx = remaining["composite_score"].nsmallest(initial_negatives).index
            y.loc[neg_idx] = -1

    labeled_idx = y[y != 0].index
    X_tr = X[labeled_idx]
    y_tr = y.loc[labeled_idx].values

    # ----- Train bagged RF -----
    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr)

    base = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf = BaggingClassifier(estimator=base, n_estimators=100, random_state=random_state)
    clf.fit(X_tr_std, y_tr)

    X_std = scaler.transform(X)
    df["predicted_probability"] = clf.predict_proba(X_std)[:, 1]

    # ----- Bait-specific decoy shuffles for FDR -----
    decoys = []
    for i, bait in enumerate(df["Bait"].unique()):
        rng_i = np.random.RandomState(random_state + i)
        sub = df[df["Bait"] == bait].copy()
        dec = sub[feature_columns].apply(lambda col: rng_i.permutation(col.values))
        X_dec = scaler.transform(dec.values)
        decoys.append(clf.predict_proba(X_dec)[:, 1])
    decoy_probs = np.concatenate(decoys) if len(decoys) else np.array([])

    real_probs = df["predicted_probability"].values
    unique_p = np.unique(real_probs)

    # raw FDR
    raw_fdr = {}
    for p in unique_p:
        n_real = np.sum(real_probs >= p)
        n_dec = np.sum(decoy_probs >= p) if decoy_probs.size else 0
        raw_fdr[p] = min(n_dec / n_real if n_real > 0 else 1.0, 1.0)

    # monotonic FDR (non-increasing with probability)
    sorted_p = np.sort(unique_p)
    mono_fdr = {sorted_p[0]: raw_fdr[sorted_p[0]]}
    for i in range(1, len(sorted_p)):
        p = sorted_p[i]
        prev = sorted_p[i - 1]
        mono_fdr[p] = min(raw_fdr[p], mono_fdr[prev])

    df["FDR"] = df["predicted_probability"].map(mono_fdr)

    # ----- Background flag by global CV (optional) -----
    if "global_cv" in df.columns:
        cv_thresh = np.nanpercentile(df["global_cv"], 25)
        df["global_cv_flag"] = df["global_cv"].apply(
            lambda v: "likely background" if pd.notna(v) and v <= cv_thresh else ""
        )
    else:
        df["global_cv_flag"] = ""

    # ===== AGGREGATE TO PROTEIN-LEVEL (per bait) =====
    # Choose aggregation for probabilities
    prob_agg = "max" if aggregate_strategy.lower() == "max" else "mean"

    grp = df.groupby(["Bait", "Protein"])
    protein_df = grp.agg(
        predicted_probability=("predicted_probability", prob_agg),
        FDR=("FDR", "min"),
        n_peptides=("Peptide", "count"),
        n_background=("global_cv_flag", lambda x: (x == "likely background").sum()),
        mean_cv=("global_cv", "mean"),
    ).reset_index()

    # Protein-level background flag (≥50% peptides flagged)
    protein_df["background_flag_protein"] = np.where(
        protein_df["n_background"] >= 0.5 * protein_df["n_peptides"],
        "likely background",
        "",
    )

    # ----- Save both -----
    os.makedirs(save_dir, exist_ok=True)
    peptide_path = os.path.join(save_dir, peptide_out)
    protein_path = os.path.join(save_dir, protein_out)

    df.to_csv(peptide_path, index=False)
    protein_df.to_csv(protein_path, index=False)

    return df, protein_df








def run_train_and_fdr_fragment(
    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
    random_state: int = 42,
    save_dir: str = ".",
    fragment_out: str = "whisper_fragment_scores.csv",
    protein_out: str = "whisper_protein_scores_from_fragments.csv",
    aggregate_strategy: str = "max",   # "max" or "mean" for fragment->protein prob aggregation
):
    """
    Train a model on FRAGMENT-level features, compute bait-specific decoy FDR,
    and aggregate to PROTEIN-level scores per bait.

    Expected columns in `features_df`:
      - Bait, Protein, Peptide, Fragment
      - composite_score, global_cv (optional), single_rep_flag (optional)
      - Feature columns:
          ['log_fold_change','snr','mean_diff','median_diff',
           'replicate_fold_change_sd','bait_cv','bait_control_sd_ratio','zero_or_neg_fc']

    Saves:
      - <save_dir>/<fragment_out>: fragment-level scores with FDR
      - <save_dir>/<protein_out>: protein-level aggregation per bait

    Returns:
      (fragment_df, protein_df)
    """
    rng = np.random.RandomState(random_state)

    # Stable order
    df = (
        features_df.copy()
        .sort_values(["Bait", "Protein", "Peptide", "Fragment"])
        .reset_index(drop=True)
    )

    feature_columns = [
        "log_fold_change", "snr", "mean_diff", "median_diff",
        "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
        "zero_or_neg_fc",
    ]
    X = df[feature_columns].values

    # ---------- Cluster baits to identify "strong" set ----------
    bait_top50_stds = {
        b: df[df["Bait"] == b]["composite_score"].nlargest(50).std()
        for b in df["Bait"].unique()
    }
    bait_names = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values()), dtype=float).reshape(-1, 1)

    if len(bait_names) > 2:
        Z = linkage(bait_scores, method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")
    else:
        clusters = np.ones(len(bait_names), dtype=int)

    cluster_sizes = {c: int(np.sum(clusters == c)) for c in np.unique(clusters)}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in np.unique(clusters)}
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]
    strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cluster_means[c])
    strong_baits = [b for b, c in zip(bait_names, clusters) if c == strong_cluster_id]

    # ---------- Pseudo-labels ----------
    y = pd.Series(0, index=df.index)  # 0=unlabeled, 1=pos, -1=neg
    bait_pos_quota = {b: (initial_positives if b in strong_baits else 0) for b in df["Bait"].unique()}

    for bait in df["Bait"].unique():
        sub = df[df["Bait"] == bait].copy()
        n_pos = bait_pos_quota[bait]
        if n_pos > 0:
            ranked = sub.sort_values("composite_score", ascending=False)
            elig = ranked[ranked.get("single_rep_flag", 0) != 1]  # exclude single-rep spikes if present
            pos_idx = elig.index[:n_pos]
            y.loc[pos_idx] = 1

            remaining = sub.drop(index=pos_idx, errors="ignore")
            neg_idx = remaining["composite_score"].nsmallest(initial_negatives).index
            y.loc[neg_idx] = -1

    labeled_idx = y[y != 0].index
    X_tr = X[labeled_idx]
    y_tr = y.loc[labeled_idx].values

    # ---------- Train bagged RF ----------
    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr)

    base = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf = BaggingClassifier(estimator=base, n_estimators=100, random_state=random_state)
    clf.fit(X_tr_std, y_tr)

    X_std = scaler.transform(X)
    df["predicted_probability"] = clf.predict_proba(X_std)[:, 1]

    # ---------- Bait-specific decoy shuffles for FDR ----------
    decoys = []
    for i, bait in enumerate(df["Bait"].unique()):
        rng_i = np.random.RandomState(random_state + i)
        sub = df[df["Bait"] == bait].copy()
        dec = sub[feature_columns].apply(lambda col: rng_i.permutation(col.values))
        X_dec = scaler.transform(dec.values)
        decoys.append(clf.predict_proba(X_dec)[:, 1])
    decoy_probs = np.concatenate(decoys) if len(decoys) else np.array([])

    real_probs = df["predicted_probability"].values
    unique_p = np.unique(real_probs)

    # raw FDR
    raw_fdr = {}
    for p in unique_p:
        n_real = np.sum(real_probs >= p)
        n_dec = np.sum(decoy_probs >= p) if decoy_probs.size else 0
        raw_fdr[p] = min(n_dec / n_real if n_real > 0 else 1.0, 1.0)

    # monotone FDR (non-increasing with prob)
    sorted_p = np.sort(unique_p)
    mono_fdr = {sorted_p[0]: raw_fdr[sorted_p[0]]}
    for i in range(1, len(sorted_p)):
        p = sorted_p[i]
        prev = sorted_p[i - 1]
        mono_fdr[p] = min(raw_fdr[p], mono_fdr[prev])

    df["FDR"] = df["predicted_probability"].map(mono_fdr)

    # ---------- Background flag by global CV (optional) ----------
    if "global_cv" in df.columns:
        cv_thresh = np.nanpercentile(df["global_cv"], 25)
        df["global_cv_flag"] = df["global_cv"].apply(
            lambda v: "likely background" if pd.notna(v) and v <= cv_thresh else ""
        )
    else:
        df["global_cv_flag"] = ""

    # ===== AGGREGATE TO PROTEIN-LEVEL (per bait) =====
    prob_agg = "max" if aggregate_strategy.lower() == "max" else "mean"

    grp = df.groupby(["Bait", "Protein"])
    protein_df = grp.agg(
        predicted_probability=("predicted_probability", prob_agg),
        FDR=("FDR", "min"),
        n_fragments=("Fragment", "count"),
        n_background=("global_cv_flag", lambda x: (x == "likely background").sum()),
        mean_cv=("global_cv", "mean"),
    ).reset_index()

    protein_df["background_flag_protein"] = np.where(
        protein_df["n_background"] >= 0.5 * protein_df["n_fragments"],
        "likely background",
        "",
    )

    # ----- Save both -----
    os.makedirs(save_dir, exist_ok=True)
    fragment_path = os.path.join(save_dir, fragment_out)
    protein_path = os.path.join(save_dir, protein_out)

    df.to_csv(fragment_path, index=False)
    protein_df.to_csv(protein_path, index=False)

    return df, protein_df
