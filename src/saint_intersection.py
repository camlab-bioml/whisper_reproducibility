"""Benchmark on the pairs both SAINT tools reported (Supplementary Fig. 6)."""

import os

import pandas as pd

from src import benchmark as pr


def saint_reported_pairs(cfg):
    to_gene = pr.make_to_gene(pr.load_uniprot_to_gene(cfg['gaf_file'])) if cfg['gaf'] else None
    sets = []
    for path in (cfg['saintq'], cfg['saintexpress']):
        if not path:
            continue
        d = pd.read_csv(path, sep='\t')
        prey = d['PreyGene' if 'PreyGene' in d.columns else 'Prey'].astype(str)
        if to_gene is not None:
            prey = prey.map(to_gene)
        sets.append(set(zip(d['Bait'].astype(str), prey.astype(str))))
    return set.intersection(*sets) if sets else None


def _restricted(frame, prey_col, keys):
    m = [k in keys for k in zip(frame['Bait'].astype(str), frame[prey_col].astype(str))]
    return frame.loc[m].reset_index(drop=True)


def run_dataset(cfg):
    keys = saint_reported_pairs(cfg)
    if keys is None:
        print(f'{cfg["name"]}: no SAINT table -- skipped')
        return
    cfg = dict(cfg, outdir=os.path.join(cfg['outdir'], 'intersection'))
    load_new, load_old = pr.load_new_methods, pr.load_old_methods

    def new_methods(c, k):
        features_df, frames = load_new(c, k)
        features_df = _restricted(features_df, 'Prey', keys)
        frames = {m: (_restricted(df, pc, keys), pc, sc)
                  for m, (df, pc, sc) in frames.items() if m != 'WHISPER'}
        frames['WHISPER'] = (features_df, 'Prey', 'predicted_probability')
        return features_df, frames

    def old_methods(c, features_df):
        return {m: (_restricted(df, pc, keys), pc, sc)
                for m, (df, pc, sc) in load_old(c, features_df).items()}

    pr.load_new_methods, pr.load_old_methods = new_methods, old_methods
    try:
        pr.run_dataset(cfg)
    finally:
        pr.load_new_methods, pr.load_old_methods = load_new, load_old
