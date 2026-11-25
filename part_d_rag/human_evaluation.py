
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
samples_path = RESULTS / "generation_samples.json"

with open(samples_path, "r", encoding="utf-8") as f:
    samples = __import__("json").load(f)

output_path = RESULTS / "rag_human_eval.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "question",
        "generated_answer",
        "supported_by_passages",
        "factually_correct",
        "evaluator"
    ])

    for sample in samples[:20]:
        writer.writerow([
            sample["question"],
            sample["generated_answer"],
            "",
            "",
            ""
        ])

print("Created:", output_path)
print("Questions:", min(20, len(samples)))
