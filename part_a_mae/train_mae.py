
import os
import sys
import csv
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from starter.eurosat import load_eurosat_split, patchify
from mae import MaskedAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
BATCH_SIZE = 64
ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():

    print("Device:", DEVICE)

    print("\nLoading EuroSAT training data...")

    images, labels = load_eurosat_split("train")

    print("Training images:", images.shape)

    dataset = TensorDataset(images)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    model = MaskedAutoencoder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=DEVICE.type == "cuda"
    )

    best_loss = float("inf")

    history_path = os.path.join(
        RESULTS_DIR,
        "mae_training_history.csv"
    )

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "training_loss", "time_seconds"])

    print("\nStarting MAE training...\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_images = 0

        start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_images,) in enumerate(loader):

            batch_images = batch_images.to(
                DEVICE,
                non_blocking=True
            )

            patches = patchify(batch_images)

            with torch.cuda.amp.autocast(
                enabled=DEVICE.type == "cuda"
            ):

                predictions, mask = model(patches)

                squared_error = (predictions - patches) ** 2

                masked_error = squared_error * mask.unsqueeze(-1)

                loss = masked_error.sum() / (
                    mask.sum() * patches.shape[-1]
                )

                group_start = (
                    batch_idx // ACCUMULATION_STEPS
                ) * ACCUMULATION_STEPS

                group_size = min(
                    ACCUMULATION_STEPS,
                    len(loader) - group_start
                )

                scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()

            should_update = (
                (batch_idx + 1) % ACCUMULATION_STEPS == 0
                or batch_idx + 1 == len(loader)
            )

            if should_update:

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * batch_images.size(0)
            total_images += batch_images.size(0)

        epoch_loss = total_loss / total_images
        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.6f} "
            f"Time: {epoch_time:.2f}s"
        )

        with open(history_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                epoch_loss,
                round(epoch_time, 2)
            ])

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss
                },
                os.path.join(
                    CHECKPOINT_DIR,
                    "best_mae.pth"
                )
            )

            print("Best model checkpoint saved.")

    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "final_mae.pth")
    )

    print("\nMAE training completed.")
    print("Best training loss:", best_loss)
    print("Training history:", history_path)
    print("Checkpoints:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
