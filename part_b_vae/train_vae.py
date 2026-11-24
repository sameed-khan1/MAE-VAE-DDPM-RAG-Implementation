
import os
import sys
import csv
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from starter.uhat import load_uhat_characters
from vae import VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

DATA_ROOT = os.path.join(ROOT, "starter", "data", "uhat", "data")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_loss(reconstruction, images, mu, logvar):

    reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstruction,
        images,
        reduction="none"
    ).sum(dim=(1, 2, 3)).mean()

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    ).mean()

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss


def run_epoch(model, loader, optimizer=None):

    training = optimizer is not None
    model.train(training)

    total_loss = 0
    total_reconstruction = 0
    total_kl = 0
    total_images = 0

    for (images,) in loader:

        images = images.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            reconstruction, mu, logvar = model(images)

            loss, rec_loss, kl_loss = calculate_loss(
                reconstruction,
                images,
                mu,
                logvar
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_reconstruction += rec_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_images += batch_size

    return (
        total_loss / total_images,
        total_reconstruction / total_images,
        total_kl / total_images
    )


def main():

    print("Device:", DEVICE)

    print("\nLoading UHaT dataset...")

    train_images, _, class_names = load_uhat_characters(
        root=DATA_ROOT,
        split="train"
    )

    test_images, _, _ = load_uhat_characters(
        root=DATA_ROOT,
        split="test",
        class_names=class_names
    )

    print("Training images:", train_images.shape)
    print("Testing images:", test_images.shape)
    print("Character classes:", len(class_names))

    train_loader = DataLoader(
        TensorDataset(train_images),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    test_loader = DataLoader(
        TensorDataset(test_images),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    model = VAE(latent_dim=16).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "vae_training_history.csv"
    )

    best_test_loss = float("inf")

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_total",
            "train_reconstruction",
            "train_kl",
            "test_total",
            "test_reconstruction",
            "test_kl",
            "time_seconds"
        ])

    print("\nStarting VAE training...\n")

    for epoch in range(EPOCHS):

        start_time = time.time()

        train_total, train_rec, train_kl = run_epoch(
            model,
            train_loader,
            optimizer
        )

        test_total, test_rec, test_kl = run_epoch(
            model,
            test_loader
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train: {train_total:.2f} "
            f"(Recon: {train_rec:.2f}, KL: {train_kl:.2f}) | "
            f"Test: {test_total:.2f} "
            f"(Recon: {test_rec:.2f}, KL: {test_kl:.2f}) | "
            f"Time: {epoch_time:.2f}s"
        )

        with open(history_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_total,
                train_rec,
                train_kl,
                test_total,
                test_rec,
                test_kl,
                epoch_time
            ])

        if test_total < best_test_loss:

            best_test_loss = test_total

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": best_test_loss
                },
                os.path.join(CHECKPOINT_DIR, "vae_best.pth")
            )

            print("Best VAE checkpoint saved.")

    print("\nVAE training completed.")
    print("Best test loss:", best_test_loss)
    print("History:", history_path)
    print("Checkpoint:", os.path.join(CHECKPOINT_DIR, "vae_best.pth"))


if __name__ == "__main__":
    main()
