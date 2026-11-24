
import os
import sys
import csv
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from starter.eurosat import (
    load_eurosat_split,
    few_label_indices,
    patchify,
    CLASS_NAMES
)
from mae import MaskedAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "best_mae.pth"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_features(model, images):

    features = []

    model.eval()

    with torch.no_grad():

        for start in range(0, len(images), BATCH_SIZE):

            batch = images[start:start + BATCH_SIZE].to(DEVICE)

            patches = patchify(batch)

            batch_features = model.encode_all(patches)

            features.append(batch_features.cpu())

    return torch.cat(features, dim=0)


def main():

    print("Device:", DEVICE)

    print("\nLoading pretrained MAE...")

    mae = MaskedAutoencoder().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    mae.load_state_dict(checkpoint["model_state_dict"])

    for parameter in mae.parameters():
        parameter.requires_grad = False

    print("MAE checkpoint loaded.")
    print("Encoder frozen.")

    print("\nLoading EuroSAT data...")

    x_train, y_train = load_eurosat_split("train")
    x_test, y_test = load_eurosat_split("test")

    few_indices = few_label_indices(
        y_train,
        per_class=16,
        seed=42
    )

    x_few = x_train[few_indices]
    y_few = y_train[few_indices]

    print("Few-label images:", len(x_few))
    print("Test images:", len(x_test))

    print("\nExtracting training features...")

    train_features = extract_features(mae, x_few)

    print("Training features:", train_features.shape)

    print("\nExtracting test features...")

    test_features = extract_features(mae, x_test)

    print("Test features:", test_features.shape)

    train_features = train_features.to(DEVICE)
    train_labels = y_few.to(DEVICE)

    classifier = nn.Linear(192, 10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "mae_classifier_history.csv"
    )

    best_accuracy = 0.0

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "training_loss", "test_accuracy"])

    print("\nStarting classifier training...\n")

    for epoch in range(EPOCHS):

        classifier.train()

        optimizer.zero_grad()

        outputs = classifier(train_features)

        loss = criterion(outputs, train_labels)

        loss.backward()
        optimizer.step()

        classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for start in range(0, len(test_features), BATCH_SIZE):

                features = test_features[
                    start:start + BATCH_SIZE
                ].to(DEVICE)

                labels = y_test[
                    start:start + BATCH_SIZE
                ].to(DEVICE)

                predictions = classifier(features).argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {loss.item():.6f} "
            f"Test Accuracy: {accuracy:.2f}%"
        )

        with open(history_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                loss.item(),
                accuracy
            ])

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                classifier.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "mae_classifier.pth"
                )
            )

    print("\nLoading best classifier...")

    classifier.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "mae_classifier.pth"),
            map_location=DEVICE,
            weights_only=True
        )
    )

    classifier.eval()

    confusion_matrix = torch.zeros(
        10, 10, dtype=torch.long
    )

    with torch.no_grad():

        for start in range(0, len(test_features), BATCH_SIZE):

            features = test_features[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            labels = y_test[
                start:start + BATCH_SIZE
            ]

            predictions = classifier(features).argmax(dim=1).cpu()

            for actual, predicted in zip(labels, predictions):
                confusion_matrix[actual, predicted] += 1

    final_accuracy = (
        confusion_matrix.diag().sum().item()
        / confusion_matrix.sum().item()
    ) * 100

    print("\n========== FINAL RESULTS ==========")

    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Final test accuracy: {final_accuracy:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix)

    print("\nClass Names:")
    print(CLASS_NAMES)

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "accuracy": final_accuracy,
            "confusion_matrix": confusion_matrix,
            "class_names": CLASS_NAMES
        },
        os.path.join(OUTPUT_DIR, "mae_classifier_results.pth")
    )

    with open(
        os.path.join(RESULTS_DIR, "mae_confusion_matrix.csv"),
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["Actual/Predicted"] + CLASS_NAMES)

        for i, row in enumerate(confusion_matrix.tolist()):
            writer.writerow([CLASS_NAMES[i]] + row)

    print("\nClassifier training completed.")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
