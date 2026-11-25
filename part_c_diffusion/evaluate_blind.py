
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUMAN_FILE = ROOT / "results" / "human_eval.csv"
ANSWER_FILE = ROOT / "results" / "ddpm_blind_answer_key.csv"
OUTPUT_FILE = ROOT / "results" / "ddpm_human_evaluation_results.csv"


def normalize(label):
    return re.sub(r"[^a-z0-9]", "", label.strip().lower())


with open(HUMAN_FILE, newline="", encoding="utf-8-sig") as f:
    human_rows = list(csv.DictReader(f))

with open(ANSWER_FILE, newline="", encoding="utf-8-sig") as f:
    answer_rows = list(csv.DictReader(f))

answer_key = {
    int(row["sample_id"]): normalize(row["actual_class"])
    for row in answer_rows
}

results = []
correct = 0
completed = 0
missing = 0

for row in human_rows:
    sample_id = int(row["sample_id"])
    predicted = normalize(row["predicted_class"])
    actual = answer_key.get(sample_id, "")

    if not predicted:
        missing += 1
        continue

    completed += 1
    is_correct = predicted == actual

    if is_correct:
        correct += 1

    results.append({
        "evaluator": row["evaluator"],
        "sample_id": sample_id,
        "predicted_class": row["predicted_class"],
        "actual_class": actual,
        "correct": is_correct
    })

accuracy = (correct / completed * 100) if completed else 0

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "evaluator",
            "sample_id",
            "predicted_class",
            "actual_class",
            "correct"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n========== BLIND EVALUATION RESULTS ==========")
print(f"Total samples: {len(answer_key)}")
print(f"Completed predictions: {completed}")
print(f"Missing predictions: {missing}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {completed - correct}")
print(f"Recognition accuracy: {accuracy:.2f}%")
print(f"\nDetailed results saved to: {OUTPUT_FILE}")
