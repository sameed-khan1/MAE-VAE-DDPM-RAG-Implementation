
from datasets import load_dataset
from config import BIOASQ_DATASET, CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_and_chunk_corpus():
    dataset = load_dataset(BIOASQ_DATASET, "text-corpus")
    abstracts = dataset["test"]

    all_chunks = []

    for abstract in abstracts:
        abstract_id = abstract["id"]
        text = abstract["passage"]

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "abstract_id": abstract_id,
                "chunk_id": index,
                "text": chunk
            })

    print(f"Total abstracts: {len(abstracts)}")
    print(f"Total chunks: {len(all_chunks)}")
    print("\nSample chunk:")
    print(all_chunks[0])

    return all_chunks


if __name__ == "__main__":
    load_and_chunk_corpus()
