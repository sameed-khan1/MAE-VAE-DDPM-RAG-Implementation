
import os
import torch

from eurosat import (
    load_eurosat_split,
    few_label_indices,
    patchify,
    unpatchify,
    CLASS_NAMES
)

from uhat import load_uhat_characters
from bioasq import load_corpus, load_questions
from viz import save_grid


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


print("\n========== PART A: EUROSAT ==========")

x_train, y_train = load_eurosat_split("train")
x_test, y_test = load_eurosat_split("test")

few = few_label_indices(y_train)
patches = patchify(x_train[:8])

print("EuroSAT train:", tuple(x_train.shape))
print("EuroSAT train labels:", tuple(y_train.shape))
print("EuroSAT test:", tuple(x_test.shape))
print("EuroSAT test labels:", tuple(y_test.shape))
print("Few-label subset:", len(few))
print("Patches:", tuple(patches.shape))

assert torch.allclose(unpatchify(patches), x_train[:8])

save_grid(
    x_train[:32],
    os.path.join(RESULTS_DIR, "eurosat_examples.png"),
    nrow=8,
    titles=[CLASS_NAMES[i] for i in y_train[:32]]
)

print("EuroSAT example grid saved.")


print("\n========== PART B AND C: UHAT ==========")

uhat_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "uhat",
    "data"
)

characters_train, labels_train, class_names = load_uhat_characters(
    root=uhat_root,
    split="train"
)

characters_test, labels_test, _ = load_uhat_characters(
    root=uhat_root,
    split="test",
    class_names=class_names
)

print("UHaT train:", tuple(characters_train.shape))
print("UHaT train labels:", tuple(labels_train.shape))
print("UHaT test:", tuple(characters_test.shape))
print("UHaT test labels:", tuple(labels_test.shape))
print("Number of character classes:", len(class_names))
print("Class names:", class_names)

if len(characters_train) >= 64:
    selected = torch.randperm(len(characters_train))[:64]
else:
    selected = torch.arange(len(characters_train))

save_grid(
    characters_train[selected],
    os.path.join(RESULTS_DIR, "uhat_examples.png"),
    nrow=8,
    titles=[class_names[i] for i in labels_train[selected]]
)

print("UHaT example grid saved.")


print("\n========== PART D: BIOASQ ==========")

corpus = load_corpus()
questions = load_questions("test", corpus)

print("BioASQ corpus:", len(corpus), "abstracts")
print("BioASQ test questions:", len(questions))

if questions:
    print("Example question:", questions[0]["question"])
    print("Relevant abstract IDs:", questions[0]["relevant_ids"][:5])

print("\n========== STARTER CHECK COMPLETED ==========")
