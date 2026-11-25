
import json
import torch
from pathlib import Path

from chunking import load_and_chunk_corpus
from retriever import Retriever
from config import RESULTS_DIR


def build_index():
    chunks = load_and_chunk_corpus()

    retriever = Retriever()

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")
    embeddings = retriever.encode(texts, batch_size=64)

    index_path = RESULTS_DIR / "bioasq_index.pt"
    chunks_path = RESULTS_DIR / "bioasq_chunks.json"

    torch.save(embeddings, index_path)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print("\nIndex created successfully.")
    print("Embedding shape:", embeddings.shape)
    print("Index saved:", index_path)
    print("Chunks saved:", chunks_path)


if __name__ == "__main__":
    build_index()
