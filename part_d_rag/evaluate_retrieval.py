
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from search import BioASQRetriever
from config import BIOASQ_DATASET, RESULTS_DIR


def evaluate():
    dataset = load_dataset(
        BIOASQ_DATASET,
        "question-answer-passages"
    )["test"]

    retriever = BioASQRetriever()

    recalls = {k: [] for k in [1, 5, 10]}
    hits = {k: [] for k in [1, 5, 10]}
    reciprocal_ranks = []

    for item in tqdm(dataset, desc="Evaluating retrieval"):
        question = item["question"]
        relevant_ids = set(item["relevant_passage_ids"])

        results = retriever.search(question, top_k=10)

        ranked_ids = [r["abstract_id"] for r in results]

        for k in [1, 5, 10]:
            retrieved = ranked_ids[:k]
            relevant_found = len(set(retrieved) & relevant_ids)

            recall = relevant_found / min(k, len(relevant_ids))
            hit = 1 if relevant_found > 0 else 0

            recalls[k].append(recall)
            hits[k].append(hit)

        reciprocal_rank = 0

        for rank, abstract_id in enumerate(ranked_ids, start=1):
            if abstract_id in relevant_ids:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    metrics = {
        "Recall@1": float(np.mean(recalls[1])),
        "Recall@5": float(np.mean(recalls[5])),
        "Recall@10": float(np.mean(recalls[10])),
        "Hit@1": float(np.mean(hits[1])),
        "Hit@5": float(np.mean(hits[5])),
        "Hit@10": float(np.mean(hits[10])),
        "MRR@10": float(np.mean(reciprocal_ranks))
    }

    print("\n========== RETRIEVAL RESULTS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    output_path = RESULTS_DIR / "retrieval_metrics.json"

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    evaluate()
