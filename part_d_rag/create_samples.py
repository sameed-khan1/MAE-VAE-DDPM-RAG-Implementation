
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

with open(RESULTS / "generation_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

idk_samples = [
    s for s in samples
    if "i don't know" in s["generated_answer"].lower()
]

non_idk = [
    s for s in samples
    if "i don't know" not in s["generated_answer"].lower()
]

best_samples = sorted(
    non_idk,
    key=lambda x: x["rougeL"],
    reverse=True
)[:3]

citation_failure = next(
    (
        s for s in non_idk
        if s["citation_precision"] == 0
    ),
    None
)

selected = best_samples.copy()

if citation_failure and citation_failure not in selected:
    selected.append(citation_failure)

if idk_samples:
    selected.append(idk_samples[0])

output = [
    "# BioASQ RAG — Qualitative Examples",
    "",
    "Five examples selected from the 100-question generation evaluation.",
    ""
]

for i, sample in enumerate(selected, 1):
    output.extend([
        f"## Example {i}",
        "",
        f"**Question:** {sample['question']}",
        "",
        f"**Reference Answer:** {sample['reference_answer']}",
        "",
        f"**Generated Answer:** {sample['generated_answer']}",
        "",
        f"**ROUGE-L:** {sample['rougeL']:.4f}",
        "",
        f"**Citation Precision:** {sample['citation_precision']:.4f}",
        "",
        "---",
        ""
    ])

with open(RESULTS / "samples.md", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved:", RESULTS / "samples.md")
print("Examples selected:", len(selected))
print("IDK examples available:", len(idk_samples))
