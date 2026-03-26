import json
import os
from Bio import Entrez
from config import ABSTRACTS_FILE, DATA_DIR

# Entrez email
# Set your email in a .env file or as an environment variable:
# export ENTREZ_EMAIL="your@email.com"
# Falls back to a placeholder if not set — NCBI requires a valid email.
Entrez.email = os.environ.get("ENTREZ_EMAIL", "your@email.com")


def fetch_pubmed(disease="diabetes", max_results=50):
    """
    Fetch PubMed abstracts for a given disease term and save to disk.

    Args:
        disease:     Search term for PubMed (default: "diabetes")
        max_results: Maximum number of abstracts to fetch (default: 50)
    """

    # Search for PubMed IDs
    print(f"Searching PubMed for: '{disease}' (max {max_results} results)...")
    try:
        handle = Entrez.esearch(db="pubmed", term=disease, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
    except Exception as e:
        raise RuntimeError(f"Failed to search PubMed: {str(e)}")

    ids = record["IdList"]
    print(f"Found {len(ids)} PubMed IDs.")

    if not ids:
        print("No results found. Try a different search term.")
        return

    # Batch fetch abstracts
    print("Fetching abstracts in batch...")
    try:
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(ids),
            rettype="abstract",
            retmode="text"
        )
        raw_text = fetch_handle.read()
        fetch_handle.close()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch abstracts: {str(e)}")

    # Split batch response into individual abstracts
    raw_abstracts = [a.strip() for a in raw_text.strip().split("\n\n\n") if a.strip()]

    abstracts = []
    for pmid, abstract_text in zip(ids, raw_abstracts):
        abstracts.append({
            "pmid": pmid,
            "abstract": abstract_text
        })

    # Save to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(ABSTRACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2)

    print(f"Saved {len(abstracts)} abstracts to {ABSTRACTS_FILE}")


if __name__ == "__main__":
    fetch_pubmed("diabetes", 50)