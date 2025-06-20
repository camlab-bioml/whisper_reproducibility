import pandas as pd
from math import log
from goatools.obo_parser import GODag

# === Constants ===
BROAD_TERMS = set([
    'GO:0005575',  # Cellular component
    'GO:0005623',  # Cell
    'GO:0005576',  # Extracellular region
    'GO:0005622',  # Intracellular
    'GO:0043226',  # Organelle
    'GO:0005624',  # Membrane
])


def parse_go_cc_annotations(gaf_file: str) -> pd.DataFrame:
    gaf_cols = [
        'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID', 'DB_Reference',
        'Evidence_Code', 'With_or_From', 'Aspect', 'DB_Object_Name', 'DB_Object_Synonym',
        'DB_Object_Type', 'Taxon_ID', 'Date', 'Assigned_By', 'Annotation_Extension',
        'Gene_Product_Form_ID'
    ]
    df = pd.read_csv(gaf_file, sep='\t', comment='!', names=gaf_cols, dtype=str)
    return df[(df['Aspect'] == 'C') & (~df['GO_ID'].isin(BROAD_TERMS))][['DB_Object_Symbol', 'GO_ID']]


def get_terms_with_parents(go_terms, go_dag):
    expanded = set()
    for term in go_terms:
        if term in go_dag:
            expanded.update({term})
            expanded.update(go_dag[term].get_all_parents())
    return expanded - BROAD_TERMS


def create_go_cc_gold_standard(go_cc: pd.DataFrame, bait: str, go_dag: GODag, term_threshold=500):
    bait_terms = go_cc[go_cc['DB_Object_Symbol'] == bait]['GO_ID'].unique()
    expanded_terms = get_terms_with_parents(bait_terms, go_dag)

    term_counts = go_cc['GO_ID'].value_counts()
    filtered_terms = [term for term in expanded_terms if term_counts.get(term, 0) <= term_threshold]

    shared_genes = go_cc[go_cc['GO_ID'].isin(filtered_terms)]['DB_Object_Symbol'].unique()
    prey_set = set(shared_genes)
    prey_set.discard(bait)

    return prey_set


def create_specific_go_list(go_cc: pd.DataFrame, bait: str, go_dag: GODag, top_n_terms=3):
    bait_terms = go_cc[go_cc['DB_Object_Symbol'] == bait]['GO_ID'].unique()
    expanded_terms = get_terms_with_parents(bait_terms, go_dag)

    term_to_proteins = {
        term: set(go_cc[go_cc['GO_ID'] == term]['DB_Object_Symbol'].unique())
        for term in expanded_terms
    }
    term_to_proteins = {term: ps for term, ps in term_to_proteins.items() if len(ps) >= 5}

    total_annotations = len(go_cc)
    term_scores = []
    for term, proteins in term_to_proteins.items():
        freq = len(proteins) / total_annotations
        ic = -log(freq) if freq > 0 else 0
        depth = go_dag[term].depth if term in go_dag else 0
        specificity = ic * depth
        term_scores.append((term, specificity, len(proteins), proteins))

    top_terms = sorted(term_scores, key=lambda x: -x[1])[:top_n_terms]

    print(f"\nTop {top_n_terms} GO:CC terms for bait {bait}:")
    for term, score, count, _ in top_terms:
        name = go_dag[term].name if term in go_dag else ''
        print(f"{term} ({name}): {count} proteins, specificity score = {score:.2f}")

    union_preys = set().union(*[p for _, _, _, p in top_terms])
    union_preys.discard(bait)

    return union_preys


def generate_gocc_ground_truth(
    gaf_file: str,
    obo_file: str,
    baits: list[str],
    output_file: str,
    method: str = "large",
    term_threshold: int = 1000,
    top_n_terms: int = 3
):
    """
    Generate GO:CC-based gold standard for a list of baits.

    method: "large" → shared terms, "small" → top N most specific terms
    """
    go_cc = parse_go_cc_annotations(gaf_file)
    go_dag = GODag(obo_file)

    all_dfs = []

    for bait in baits:
        print(f"\nProcessing bait: {bait}")

        if method == "large":
            prey_set = create_go_cc_gold_standard(go_cc, bait, go_dag, term_threshold)
        elif method == "small":
            prey_set = create_specific_go_list(go_cc, bait, go_dag, top_n_terms)
        else:
            raise ValueError("Invalid method. Use 'large' or 'small'.")

        df = pd.DataFrame({'Bait': bait, 'Prey': list(prey_set)})
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"\nSaved GO:CC interactions to {output_file}")
