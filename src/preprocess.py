import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import ABSTRACTS_FILE, CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text):
    """
    Light cleaning of raw PubMed abstract text.
    - Collapses multiple whitespace/newlines into a single space
    - Strips leading/trailing whitespace
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess():
    """
    Load raw PubMed abstracts, clean and chunk them, and save to disk.
    Each chunk is stored as a dict with 'text' and 'pmid' keys so that
    source attribution is preserved through the full pipeline.
    """

    # Load abstracts
    if not ABSTRACTS_FILE.exists():
        raise FileNotFoundError(
            f"Abstracts file not found at {ABSTRACTS_FILE}. "
            "Please run fetch_data.py first."
        )

    with open(ABSTRACTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total papers loaded: {len(data)}")

    # Extract and clean abstracts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    abstract_count = 0

    for item in data:
        if "abstract" in item and item["abstract"]:
            abstract_count += 1
            cleaned = clean_text(item["abstract"])
            chunks = text_splitter.split_text(cleaned)

            # Attach the PMID to every chunk from this abstract
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "pmid": item["pmid"]
                })

    print(f"Total abstracts extracted: {abstract_count}")
    print(f"Total chunks created: {len(all_chunks)}")

    # Save chunks to disk
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Chunks saved to {CHUNKS_FILE}")


if __name__ == "__main__":
    preprocess()