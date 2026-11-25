
from datasets import load_dataset
from config import BIOASQ_DATASET


def load_bioasq():
    dataset = load_dataset(
        BIOASQ_DATASET,
        "question-answer-passages"
    )

    print("\nDataset splits:")
    for split in dataset:
        print(f"{split}: {len(dataset[split])} records")

    print("\nSample training record:")
    print(dataset["train"][0])

    return dataset


if __name__ == "__main__":
    load_bioasq()
