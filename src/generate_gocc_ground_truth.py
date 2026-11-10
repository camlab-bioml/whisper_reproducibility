"""GO:CC reference."""

import json

import pandas as pd

EXCLUDE_TERMS = {
    'GO:0005575',  # cellular component
    'GO:0005623',  # cell
    'GO:0005576',  # extracellular region
    'GO:0005622',  # intracellular
    'GO:0043226',  # organelle
    'GO:0005624',  # membrane
}
GAF_COLUMNS = ['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
               'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect',
               'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 'Taxon_ID',
               'Date', 'Assigned_By', 'Annotation_Extension', 'Gene_Product_Form_ID']


def generate_gocc_ground_truth(baits, output_file, gaf_file, obo_file,
                               term_size_threshold=1000, bait_self_prey=None,
                               cache_file='data/go_term_members_cache.json'):
    from goatools.obo_parser import GODag

    bait_self_prey = bait_self_prey or {}
    dag = GODag(obo_file)
    with open(cache_file) as fh:
        cache = json.load(fh)
    gp = None
    gaf = pd.read_csv(gaf_file, sep='\t', comment='!', names=GAF_COLUMNS, dtype=str)
    go_cc = gaf[(gaf['Aspect'] == 'C') & (~gaf['GO_ID'].isin(EXCLUDE_TERMS))]

    def members(go_id):
        nonlocal gp
        if go_id not in cache:
            from gprofiler import GProfiler
            gp = gp or GProfiler(return_dataframe=True)
            try:
                res = gp.convert(organism='hsapiens', query=go_id, target_namespace='ENTREZGENE')
                cache[go_id] = sorted(set(res['name'].dropna().unique())) if not res.empty else []
            except Exception:
                return set()
        return set(cache[go_id])

    frames = []
    for bait in baits:
        terms = set(go_cc[go_cc['DB_Object_Symbol'] == bait]['GO_ID'].dropna().unique())
        expanded = set()
        for t in terms:
            if t in dag:
                expanded.add(t)
                expanded.update(dag[t].get_all_parents())
        preys = set()
        for term in sorted(t for t in expanded if t not in EXCLUDE_TERMS):
            m = members(term)
            m.discard(bait)
            if 1 < len(m) <= term_size_threshold:
                preys.update(m)
        preys.update(bait_self_prey.get(bait, [bait]))
        frames.append(pd.DataFrame({'Bait': bait, 'Prey': sorted(preys)}))
    with open(cache_file, 'w') as fh:
        json.dump(cache, fh)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_file, index=False)
    return out
