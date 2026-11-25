
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

path = RESULTS / "rag_human_eval.csv"

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

total = len(rows)

supported = sum(
    row["supported_by_passages"].strip().lower() == "yes"
    for row in rows
)

correct = sum(
    row["factually_correct"].strip().lower() == "yes"
    for row in rows
)

metrics = {
    "total_evaluated": total,
    "supported_answers": supported,
    "unsupported_answers": total - supported,
    "factual_correct_answers": correct,
    "factual_incorrect_answers": total - correct,
    "support_rate": supported / total,
    "factual_accuracy": correct / total
}

with open(RESULTS / "rag_human_evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n========== RAG HUMAN EVALUATION ==========")
print("Total evaluated:", total)
print(f"Supported answers: {supported}/{total} ({metrics['support_rate']:.2%})")
print(f"Factually correct: {correct}/{total} ({metrics['factual_accuracy']:.2%})")
print("\nSaved: results/rag_human_evaluation_metrics.json")
