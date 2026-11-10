"""WHISPER feature engineering."""

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "log_fold_change", "snr", "mean_diff", "median_diff",
    "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
    "zero_or_neg_fc",
]
HEURISTIC_FEATURES = ["log_fold_change", "snr", "mean_diff", "median_diff"]

ID_COLUMNS = {
    "protein": ["Protein"],
    "peptide": ["Protein", "Peptide"],
    "fragment": ["Protein", "Peptide", "Fragment"],
}


def read_matrix(path, level="protein"):
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    return df.rename(columns={"protein": "Protein", "peptide": "Peptide"})


def _unnamed_rows(df, level):
    bad = {"", "nan", "None"} | ({"0"} if level != "protein" else set())
    return df["Protein"].isna() | df["Protein"].astype(str).str.strip().isin(bad)


def _control_columns(columns, controls, match):
    if match == "exact":
        return [c for c in columns if re.sub(r"_\d+$", "", c) in controls]
    return [c for c in columns if any(ctrl in c for ctrl in controls)]


def run_feature_engineering(intensity_df, controls, baits=None, level="protein",
                            control_match="substring", labelling_enzyme="birA",
                            bait_self_protein=None, drop_unnamed=False):

    id_cols = ID_COLUMNS[level]
    bait_self_protein = bait_self_protein or {}
    df = intensity_df.copy()

    if drop_unnamed:
        unnamed = _unnamed_rows(df, level)
        if unnamed.any():
            print(f"Dropping {int(unnamed.sum())} row(s) with no protein identifier")
            df = df.loc[~unnamed].reset_index(drop=True)
    dups = df.duplicated(subset=id_cols, keep=False)
    if dups.any():
        print(f"WARNING: {int(dups.sum())} rows share duplicated {id_cols} keys")

    sample_cols = [c for c in df.columns if c not in id_cols]
    control_cols = _control_columns(df.columns, controls, control_match)
    if baits is None:
        baits = sorted({c.split("_")[0] for c in sample_cols
                        if c not in control_cols})

    # global CV per identifier across every sample column
    values = df[sample_cols].astype(float).to_numpy()
    means = values.mean(axis=1)
    sds = values.std(axis=1)
    cv = np.where(means > 0, sds / np.where(means > 0, means, 1), 0)
    keys = [tuple(r) if len(id_cols) > 1 else r[0]
            for r in df[id_cols].itertuples(index=False, name=None)]
    global_cv = dict(zip(keys, cv))

    frames = []
    for bait in baits:
        if level == "protein":
            bait_cols = [c for c in df.columns if re.fullmatch(fr"{re.escape(bait)}_\d+", c)]
        else:
            bait_cols = [f"{bait}_1", f"{bait}_2", f"{bait}_3"]
            if not all(c in df.columns for c in bait_cols):
                print(f"[WARN] Skipping bait {bait}: replicate columns missing")
                continue
        frames.append(_bait_features(df, bait, bait_cols, control_cols, id_cols,
                                     level, labelling_enzyme, bait_self_protein,
                                     global_cv))
    out = pd.concat(frames, ignore_index=True)
    if level == "protein":
        out = out.rename(columns={"Protein": "Prey"})
    return out


def _bait_features(df, bait, bait_cols, control_cols, id_cols, level,
                   labelling_enzyme, bait_self_protein, global_cv):
    inert_names = {bait, bait_self_protein.get(bait, bait), labelling_enzyme} - {None}
    is_inert = df["Protein"].astype(str).map(
        lambda p: bool(inert_names & set(p.split(";"))))
    eligible_df = df[~is_inert]
    inert_df = df[is_inert]

    ctrl = eligible_df[control_cols].astype(float).to_numpy()
    ctrl_means = ctrl.mean(axis=1)
    ctrl_sds = ctrl.std(axis=1)
    # floors for an all-zero control row; the protein pipeline falls back to
    # 1000 and the peptide/fragment pipelines to 1.0 when no control is nonzero
    fallback = 1000 if level == "protein" else 1.0
    min_mean = ctrl_means[ctrl_means > 0].min() if (ctrl_means > 0).any() else fallback
    min_sd = ctrl_sds[ctrl_sds > 0].min() if (ctrl_sds > 0).any() else fallback

    rows = []
    for source, inert_flag in ((eligible_df, False), (inert_df, True)):
        for _, row in source.iterrows():
            b = row[bait_cols].astype(float).values
            c = row[control_cols].astype(float).values
            mean_b, median_b, sd_b = np.mean(b), np.median(b), np.std(b)
            mean_c, sd_c = np.mean(c), np.std(c)
            mean_c = mean_c if mean_c > 0 else min_mean
            sd_c = sd_c if sd_c > 0 else min_sd

            zero_count = np.sum(b == 0)
            log_fc = np.log2(mean_b / mean_c + 1e-5) / max(1, zero_count)
            snr = (mean_b / sd_c) / max(1, zero_count)
            nonzero_reps = int(np.sum(b > 0))

            rec = {"Bait": bait}
            for col in id_cols:
                rec[col] = row[col]
            rec.update({
                "log_fold_change": log_fc,
                "snr": snr,
                "mean_diff": mean_b - mean_c,
                "median_diff": median_b - np.median(c),
                "replicate_fold_change_sd": np.std(b / mean_c),
                "bait_cv": sd_b / mean_b if mean_b != 0 else 0,
                "bait_control_sd_ratio": sd_b / sd_c,
                "zero_or_neg_fc": 0 if log_fc <= 0 else 1,
                "nonzero_reps": nonzero_reps,
                "reps_above_ctrl_med": int(np.sum(b > np.median(c))),
                "single_rep_flag": 1 if nonzero_reps == 1 else 0,
                "is_inert": inert_flag,
            })
            rows.append(rec)
    feat = pd.DataFrame(rows)

    # Heuristic score: features z-scored within the bait (fit on the eligible
    # rows, applied to all) and averaged over the four enrichment features.
    eligible = ~feat["is_inert"].to_numpy()
    scaler = StandardScaler().fit(feat.loc[eligible, FEATURE_COLUMNS])
    scaled = pd.DataFrame(scaler.transform(feat[FEATURE_COLUMNS]),
                          columns=FEATURE_COLUMNS, index=feat.index)
    feat["composite_score"] = scaled[HEURISTIC_FEATURES].sum(axis=1) / len(HEURISTIC_FEATURES)
    if len(id_cols) > 1:
        feat["global_cv"] = [global_cv[tuple(k)] for k in
                             feat[id_cols].itertuples(index=False, name=None)]
    else:
        feat["global_cv"] = feat[id_cols[0]].map(global_cv)
    return feat.sort_values("composite_score", ascending=False)
