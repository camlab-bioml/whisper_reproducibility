import requests
import pandas as pd
from collections import Counter
from time import sleep


def get_biogrid_interactions(bait: str, access_key: str = "4e66c14d9e05dacb893c9c3259131ef1") -> pd.DataFrame:
    """
    Query BioGRID for physical interactions of a given gene symbol.

    Parameters:
        bait (str): Official gene symbol.
        access_key (str): BioGRID API access key.

    Returns:
        pd.DataFrame: DataFrame with columns ['Bait', 'Prey'].
    """
    BASE_URL = "https://webservice.thebiogrid.org/interactions"
    params = {
        "accesskey": access_key,
        "format": "json",
        "geneList": bait,
        "searchNames": "true",
        "includeInteractors": "true",
        "includeInteractorInteractions": "false",
        "taxId": 9606,
        "includeEvidence": "false"
    }

    response = requests.get(BASE_URL, params=params)
    try:
        interactions = response.json()
    except Exception as e:
        print(f"[ERROR] Failed to parse response for {bait}: {e}")
        return pd.DataFrame(columns=["Bait", "Prey"])

    df = pd.DataFrame.from_dict(interactions, orient="index")

    if df.empty or "EXPERIMENTAL_SYSTEM_TYPE" not in df:
        return pd.DataFrame(columns=["Bait", "Prey"])

    df_physical = df[df["EXPERIMENTAL_SYSTEM_TYPE"] == "physical"]
    interactors = df_physical[["OFFICIAL_SYMBOL_A", "OFFICIAL_SYMBOL_B"]]
    interactors.columns = ["InteractorA", "InteractorB"]

    combined = pd.melt(interactors, value_vars=["InteractorA", "InteractorB"],
                       var_name="Type", value_name="Interactor")
    combined = combined[combined["Interactor"] != bait]

    counts = Counter(combined["Interactor"])
    high_confidence = pd.DataFrame(list(counts.items()), columns=["Prey", "Occurrences"])
    high_confidence = high_confidence[high_confidence["Occurrences"] >= 1].sort_values("Occurrences", ascending=False)
    high_confidence["Bait"] = bait

    return high_confidence[["Bait", "Prey"]]


def generate_biogrid_ground_truth(baits: list[str], output_file: str = "biogrid_interactions.csv", access_key: str = "4e66c14d9e05dacb893c9c3259131ef1"):
    """
    Retrieve and save BioGRID physical interactions for a list of baits.

    Parameters:
        baits (list[str]): List of bait gene names.
        output_file (str): Where to save the resulting CSV.
        access_key (str): BioGRID API key.
    """
    all_dfs = []
    for bait in baits:
        print(f"[INFO] Querying BioGRID for bait: {bait}")
        try:
            df = get_biogrid_interactions(bait, access_key)
            all_dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Failed for {bait}: {e}")
        sleep(1)  # Respect rate limits

    if not all_dfs:
        print("[WARNING] No data retrieved.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"[DONE] BioGRID interactions saved to: {output_file}")
