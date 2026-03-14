from Bio import Entrez
import json
import os

Entrez.email = "tennis.vishal@gmail.com"

def fetch_pubmed(disease, max_results=50):
    handle = Entrez.esearch(db="pubmed", term=disease, retmax=max_results)
    record = Entrez.read(handle)

    ids = record["IdList"]
    abstracts = []

    for pmid in ids:
        fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract_text = fetch_handle.read()

        abstracts.append({
            "pmid": pmid,
            "abstract": abstract_text
        })

    os.makedirs("../data", exist_ok=True)

    with open(f"../data/{disease}_abstracts.json", "w") as f:
        json.dump(abstracts, f)

    print(f"Saved {len(abstracts)} abstracts")


if __name__ == "__main__":
    fetch_pubmed("diabetes", 50)