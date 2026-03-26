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
    abstracts = []
    for item in data:
        if "abstract" in item and item["abstract"]:
            cleaned = clean_text(item["abstract"])
            abstracts.append(cleaned)

    print(f"Total abstracts extracted: {len(abstracts)}")

    # Chunk abstracts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    for abstract in abstracts:
        chunks = text_splitter.split_text(abstract)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    # Save chunks to disk
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Chunks saved to {CHUNKS_FILE}")


if __name__ == "__main__":
    preprocess()