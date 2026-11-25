
import json
import torch

from retriever import Retriever
from config import RESULTS_DIR


class BioASQRetriever:
    def __init__(self):
        self.retriever = Retriever()

        self.index = torch.load(
            RESULTS_DIR / "bioasq_index.pt",
            map_location="cpu",
            weights_only=True
        )

        with open(
            RESULTS_DIR / "bioasq_chunks.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.chunks = json.load(f)

    def search(self, question, top_k=5):
        query_embedding = self.retriever.encode([question])

        similarities = torch.matmul(
            query_embedding,
            self.index.T
        ).squeeze(0)

        abstract_results = {}

        for index, score in enumerate(similarities.tolist()):
            chunk = self.chunks[index]
            abstract_id = chunk["abstract_id"]

            if (
                abstract_id not in abstract_results
                or score > abstract_results[abstract_id]["score"]
            ):
                abstract_results[abstract_id] = {
                    "abstract_id": abstract_id,
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": score
                }

        results = sorted(
            abstract_results.values(),
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]


if __name__ == "__main__":
    retriever = BioASQRetriever()

    question = "What is the implication of histone lysine methylation in medulloblastoma?"

    results = retriever.search(question, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"\nRank {i}")
        print("Abstract ID:", result["abstract_id"])
        print("Chunk ID:", result["chunk_id"])
        print("Score:", result["score"])
        print("Text:", result["text"][:300])

    print("\nUnique abstracts:", len(set(
        result["abstract_id"] for result in results
    )))
