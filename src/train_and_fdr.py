"""WHISPER training and FDR estimation (Supplementary Fig. 20)."""

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_COLUMNS

warnings.filterwarnings("ignore")


def strong_baits(df_real, df_eligible):
    stds = {b: df_eligible[df_eligible["Bait"] == b]["composite_score"].nlargest(50).std()
            for b in df_real["Bait"].unique()}
    names = np.array(list(stds.keys()))
    scores = np.array(list(stds.values())).reshape(-1, 1)
    if len(names) > 2:
        clusters = fcluster(linkage(scores, method="ward"), t=2, criterion="maxclust")
    else:
        clusters = np.ones(len(names), dtype=int)
    sizes = {c: int(np.sum(clusters == c)) for c in np.unique(clusters)}
    means = {c: float(scores[clusters == c].mean()) for c in np.unique(clusters)}
    cands = [c for c, n in sizes.items() if n == max(sizes.values())]
    strong = cands[0] if len(cands) == 1 else min(cands, key=lambda c: means[c])
    return [b for b, c in zip(names, clusters) if c == strong]


def weak_labels(df_real, df_eligible, strong, initial_positives, initial_negatives):
    y = pd.Series(0, index=df_real.index)
    for bait in df_real["Bait"].unique():
        if bait not in strong:
            continue
        bait_df = df_eligible[df_eligible["Bait"] == bait].copy()
        ranked = bait_df.sort_values("composite_score", ascending=False)
        positives = ranked[ranked["single_rep_flag"] != 1].index[:initial_positives]
        y.loc[positives] = 1
        remaining = bait_df.drop(index=positives, errors="ignore")
        y.loc[remaining["composite_score"].nsmallest(initial_negatives).index] = -1
    return y


def build_model(seed=42, n_jobs=None, rf_params=None):
    # Serial by default: parallel prediction averages the trees in a different
    # order, which moves probabilities by ~1e-16 and flips a few decoys across
    # ties, so the FDR then differs from the deposited one by up to ~1e-3.
    rf = RandomForestClassifier(random_state=seed, **{"n_estimators": 100, **(rf_params or {})})
    return BaggingClassifier(estimator=rf, n_estimators=100, random_state=seed,
                             n_jobs=n_jobs)


def monotone_q(real_probs, decoy_probs, n_replicates, query_probs):
    """Pooled target-decoy FDR with the running-minimum transform, read off
    for `query_probs` at the nearest real probability at or below each."""
    decoy = np.sort(np.asarray(decoy_probs))
    cuts = np.sort(np.unique(real_probs))
    n_real = len(real_probs) - np.searchsorted(np.sort(real_probs), cuts, side="left")
    n_decoy = (len(decoy) - np.searchsorted(decoy, cuts, side="left")) / n_replicates
    raw = np.minimum(np.where(n_real > 0, n_decoy / np.maximum(n_real, 1), 1.0), 1.0)
    q = np.minimum.accumulate(raw)
    pos = np.clip(np.searchsorted(cuts, query_probs, side="right") - 1, 0, len(cuts) - 1)
    return q[pos]


def fit(features_df, initial_positives=15, initial_negatives=200, level="protein",
        seed=42, n_jobs=None, rf_params=None):
    """Weak labels, fitted model and scores. Returns (scored table sorted by
    bait and identifier, model, scaler, weak labels)."""
    id_col = "Prey" if level == "protein" else "Protein"
    df_real = features_df.sort_values(["Bait", id_col]).reset_index(drop=True)
    if level != "protein":
        df_real = (df_real.groupby(["Bait", "Protein"])
                   .filter(lambda x: len(x) >= 2).reset_index(drop=True))
    df_eligible = df_real[~df_real["is_inert"]]
    np.random.seed(seed)

    x_real = df_real[FEATURE_COLUMNS].values
    strong = strong_baits(df_real, df_eligible)
    y = weak_labels(df_real, df_eligible, strong, initial_positives, initial_negatives)
    labeled = y[y != 0].index

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_real[labeled])
    model = build_model(seed, n_jobs=n_jobs, rf_params=rf_params)
    model.fit(x_train, y.loc[labeled])
    df_real["predicted_probability"] = model.predict_proba(scaler.transform(x_real))[:, 1]
    return df_real, model, scaler, y


def run_training_and_fdr(features_df, initial_positives=15, initial_negatives=200,
                         n_decoy_replicates=30, level="protein", seed=42, rf_params=None):
    """Score every pair and estimate its FDR.

    Returns (scored table, feature importances, protein table or None). The
    protein table is only produced for peptide- and fragment-level input.
    """
    df_real, model, scaler, _y = fit(features_df, initial_positives, initial_negatives,
                                     level=level, seed=seed, rf_params=rf_params)
    df_eligible = df_real[~df_real["is_inert"]]

    decoy_probs, decoy_rows = [], []
    for rep in range(n_decoy_replicates):
        for i, bait in enumerate(df_eligible["Bait"].unique()):
            np.random.seed(seed + rep * 1000 + i)
            decoy = df_eligible[df_eligible["Bait"] == bait].copy()
            for col in FEATURE_COLUMNS:
                decoy[col] = np.random.permutation(decoy[col].values)
            p = model.predict_proba(scaler.transform(decoy[FEATURE_COLUMNS].values))[:, 1]
            decoy_probs.append(p)
            if level != "protein":
                decoy_rows.append(pd.DataFrame({"Bait": decoy["Bait"].values,
                                                "Protein": decoy["Protein"].values,
                                                "predicted_probability": p, "rep": rep}))
    decoy_probs = np.concatenate(decoy_probs)

    df_real["FDR"] = monotone_q(df_eligible["predicted_probability"].values, decoy_probs,
                                n_decoy_replicates, df_real["predicted_probability"].values)
    cv_threshold = np.percentile(df_real["global_cv"], 25)
    df_real["global_cv_flag"] = np.where(df_real["global_cv"] <= cv_threshold,
                                         "likely background", "")

    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    importance_df = (pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
                     .sort_values("importance", ascending=True))

    protein_df = None
    if level != "protein":
        protein_df = _aggregate_to_protein(df_real, df_eligible, pd.concat(decoy_rows),
                                           n_decoy_replicates, level)
    return df_real, importance_df, protein_df


def _aggregate_to_protein(df_real, df_eligible, decoy_rows, n_replicates, level):
    unit = "Peptide" if level == "peptide" else "Fragment"
    flags = (df_real.groupby(["Bait", "Protein"])
             .agg(n_units=(unit, "count"),
                  n_background=("global_cv_flag", lambda x: (x == "likely background").sum()))
             .reset_index())
    flags["background_flag_protein"] = np.where(
        flags["n_background"] >= 0.5 * flags["n_units"], "likely background", "")

    prot = df_real.groupby(["Bait", "Protein"])["predicted_probability"].max().reset_index()
    real_max = df_eligible.groupby(["Bait", "Protein"])["predicted_probability"].max().values
    decoy_max = decoy_rows.groupby(["rep", "Bait", "Protein"])["predicted_probability"].max().values
    prot["FDR"] = monotone_q(real_max, decoy_max, n_replicates,
                             prot["predicted_probability"].values)
    return prot.merge(flags[["Bait", "Protein", "background_flag_protein"]],
                      on=["Bait", "Protein"], how="left")
