import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("../data/diabetes_abstracts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total papers loaded: {len(data)}")

abstracts = []

for item in data:
    if "abstract" in item and item["abstract"]:
        abstracts.append(item["abstract"])

print(f"Total abstracts extracted: {len(abstracts)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

all_chunks = []

for abstract in abstracts:
    chunks = text_splitter.split_text(abstract)
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")


with open("../data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print("Chunks saved to data/chunks.json")