
from datasets import load_dataset

BIOASQ_REPO = "enelpol/rag-mini-bioasq"


def load_corpus():
    dataset = load_dataset(BIOASQ_REPO, "text-corpus")
    split = list(dataset.keys())[0]

    corpus = {
        int(row["id"]): row["passage"]
        for row in dataset[split]
        if row["passage"]
    }

    return corpus


def load_questions(split="test", corpus=None):
    dataset = load_dataset(
        BIOASQ_REPO,
        "question-answer-passages",
        split=split
    )

    questions = []

    for row in dataset:
        relevant_ids = [
            int(value)
            for value in row["relevant_passage_ids"]
        ]

        if corpus is not None:
            relevant_ids = [
                value
                for value in relevant_ids
                if value in corpus
            ]

        if relevant_ids:
            questions.append({
                "qid": int(row["id"]),
                "question": row["question"],
                "answer": row["answer"],
                "relevant_ids": relevant_ids
            })

    return questions
