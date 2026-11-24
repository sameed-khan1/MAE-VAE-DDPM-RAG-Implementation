
import os
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter.uhat import load_uhat_characters
from model import DiffusionUNet
from diffusion import Diffusion


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 40
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "starter", "data", "uhat", "data")

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


print("Loading UHaT dataset...")

train_images, train_labels, class_names = load_uhat_characters(
    root=DATA_ROOT,
    split="train"
)

test_images, test_labels, _ = load_uhat_characters(
    root=DATA_ROOT,
    split="test",
    class_names=class_names
)

print("Classes:", len(class_names))
print("Training images:", len(train_images))
print("Testing images:", len(test_images))


train_images = train_images * 2.0 - 1.0
test_images = test_images * 2.0 - 1.0

train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)


model = DiffusionUNet().to(DEVICE)
diffusion = Diffusion(timesteps=1000, device=DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE == "cuda")
)

best_test_loss = float("inf")

history = []

print("Device:", DEVICE)
print("Starting DDPM training...")


for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        noise = torch.randn_like(images)
        timesteps = diffusion.sample_timesteps(images.size(0))

        noisy_images = diffusion.add_noise(
            images,
            noise,
            timesteps
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE == "cuda")
        ):
            predicted_noise = model(
                noisy_images,
                timesteps,
                labels
            ).sample

            loss = criterion(predicted_noise, noise)
            scaled_loss = loss / ACCUMULATION_STEPS

        scaler.scale(scaled_loss).backward()

        should_update = (
            (step + 1) % ACCUMULATION_STEPS == 0
            or (step + 1) == len(train_loader)
        )

        if should_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            noise = torch.randn_like(images)
            timesteps = diffusion.sample_timesteps(images.size(0))

            noisy_images = diffusion.add_noise(
                images,
                noise,
                timesteps
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(DEVICE == "cuda")
            ):
                predicted_noise = model(
                    noisy_images,
                    timesteps,
                    labels
                ).sample

                loss = criterion(predicted_noise, noise)

            test_loss += loss.item()

    test_loss /= len(test_loader)

    history.append([
        epoch + 1,
        train_loss,
        test_loss
    ])

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Test Loss: {test_loss:.6f}"
    )

    if test_loss < best_test_loss:

        best_test_loss = test_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": test_loss,
                "class_names": class_names
            },
            os.path.join(CHECKPOINT_DIR, "ddpm_best.pth")
        )

        print("Best checkpoint saved.")


with open(
    os.path.join(RESULTS_DIR, "ddpm_training_history.csv"),
    "w",
    newline=""
) as file:

    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "test_loss"])
    writer.writerows(history)


print("\nDDPM training completed.")
print("Best test loss:", best_test_loss)
