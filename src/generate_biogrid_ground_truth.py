"""BioGRID reference."""

import os
from collections import Counter

import pandas as pd

BASE_URL = 'https://webservice.thebiogrid.org/interactions'


def get_biogrid_interactions(gene, access_key):
    import requests
    params = {'accesskey': access_key, 'format': 'json', 'geneList': gene,
              'searchNames': 'true', 'includeInteractors': 'true',
              'includeInteractorInteractions': 'false', 'taxId': 9606,
              'includeEvidence': 'false'}
    interactions = requests.get(BASE_URL, params=params).json()
    df = pd.DataFrame.from_dict(dict(interactions.items()), orient='index')
    physical = df[df['EXPERIMENTAL_SYSTEM_TYPE'] == 'physical']
    pairs = physical[['OFFICIAL_SYMBOL_A', 'OFFICIAL_SYMBOL_B']]
    partners = pd.melt(pairs, value_name='Interactor')['Interactor']
    counts = Counter(partners[partners != gene])
    return pd.DataFrame(counts.items(), columns=['Prey', 'Occurrences']).sort_values(
        'Occurrences', ascending=False)


def generate_biogrid_ground_truth(baits, output_file, bait_self_prey=None, access_key=None):
    access_key = access_key or os.environ.get('BIOGRID_ACCESS_KEY')
    if not access_key:
        raise SystemExit('Set BIOGRID_ACCESS_KEY to query BioGRID.')
    bait_self_prey = bait_self_prey or {}
    frames = []
    for bait in baits:
        found = get_biogrid_interactions(bait, access_key)
        found['Bait'] = bait
        frames.append(found[['Bait', 'Prey']])
        frames.append(pd.DataFrame({'Bait': bait, 'Prey': bait_self_prey.get(bait, [bait])}))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_file, index=False)
    return out
