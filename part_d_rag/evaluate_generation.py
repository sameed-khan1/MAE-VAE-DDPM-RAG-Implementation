
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer

from search import BioASQRetriever
from generator import BioASQGenerator
from config import BIOASQ_DATASET, RESULTS_DIR


def evaluate():
    dataset = load_dataset(
        BIOASQ_DATASET,
        "question-answer-passages"
    )["test"]

    dataset = dataset.select(range(100))

    retriever = BioASQRetriever()
    generator = BioASQGenerator()

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    rouge_scores = []
    citation_precisions = []
    idk_count = 0
    samples = []

    for item in tqdm(dataset, desc="Evaluating generation"):
        question = item["question"]
        reference = item["answer"]

        retrieved = retriever.search(question, top_k=5)
        answer = generator.generate(question, retrieved)

        rouge = scorer.score(reference, answer)["rougeL"].fmeasure
        rouge_scores.append(rouge)

        cited_numbers = re.findall(r"\[(\d+)\]", answer)
        valid_citations = [
            int(number)
            for number in cited_numbers
            if 1 <= int(number) <= len(retrieved)
        ]

        if cited_numbers:
            precision = len(valid_citations) / len(cited_numbers)
        else:
            precision = 0.0

        citation_precisions.append(precision)

        if "i don't know" in answer.lower():
            idk_count += 1

        samples.append({
            "question": question,
            "reference_answer": reference,
            "generated_answer": answer,
            "rougeL": rouge,
            "citation_precision": precision
        })

    metrics = {
        "evaluated_questions": len(dataset),
        "average_rougeL": sum(rouge_scores) / len(rouge_scores),
        "average_citation_precision": sum(citation_precisions) / len(citation_precisions),
        "idk_rate": idk_count / len(dataset)
    }

    with open(RESULTS_DIR / "generation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(RESULTS_DIR / "generation_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    print("\n========== GENERATION RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nSaved generation metrics and samples.")


if __name__ == "__main__":
    evaluate()
