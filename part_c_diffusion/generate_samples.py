
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter.uhat import load_uhat_characters
from model import DiffusionUNet
from diffusion import Diffusion


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "starter", "data", "uhat", "data")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "ddpm_best.pth")
RESULTS_DIR = os.path.join(ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = DiffusionUNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

diffusion = Diffusion(timesteps=1000, device=DEVICE)

print("Device:", DEVICE)
print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Number of classes:", len(class_names))


# --------------------------------------------------
# 1. Generate 8 samples for each of the 40 classes
# --------------------------------------------------

all_samples = []

for class_id, class_name in enumerate(class_names):

    labels = torch.full(
        (8,),
        class_id,
        device=DEVICE,
        dtype=torch.long
    )

    samples = diffusion.sample(
        model,
        labels,
        image_size=28
    )

    samples = (samples + 1) / 2
    all_samples.append(samples.cpu())

    print(f"Generated samples for class {class_id}: {class_name}")


all_samples = torch.cat(all_samples, dim=0)

fig, axes = plt.subplots(40, 8, figsize=(12, 45))

for class_id in range(40):
    for sample_id in range(8):

        image = all_samples[class_id * 8 + sample_id, 0]

        axes[class_id, sample_id].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[class_id, sample_id].axis("off")

    axes[class_id, 0].set_ylabel(
        class_names[class_id],
        rotation=0,
        labelpad=45,
        fontsize=8
    )

plt.suptitle("DDPM: 8 Generated Samples per Character Class")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "ddpm_class_samples.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Class samples saved.")


# --------------------------------------------------
# 2. Forward noising visualization
# --------------------------------------------------

test_images, test_labels, _ = load_uhat_characters(
    root=DATA_ROOT,
    split="test",
    class_names=class_names
)

original = test_images[0:1].to(DEVICE) * 2 - 1

timesteps_to_show = [0, 100, 250, 500, 750, 999]

noise = torch.randn_like(original)

fig, axes = plt.subplots(1, 6, figsize=(15, 3))

for i, t in enumerate(timesteps_to_show):

    timestep = torch.tensor([t], device=DEVICE)

    noisy = diffusion.add_noise(
        original,
        noise,
        timestep
    )

    image = ((noisy[0, 0] + 1) / 2).clamp(0, 1)

    axes[i].imshow(image.cpu(), cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(f"t = {t}")
    axes[i].axis("off")

plt.suptitle("Forward Diffusion: Adding Noise")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "ddpm_forward_noising.png"),
    dpi=200
)

plt.close()

print("Forward noising visualization saved.")


# --------------------------------------------------
# 3. Reverse denoising snapshots
# --------------------------------------------------

labels = torch.zeros(8, dtype=torch.long, device=DEVICE)

x = torch.randn(8, 1, 28, 28, device=DEVICE)

snapshots = {}
snapshot_timesteps = [999, 750, 500, 250, 100, 0]

with torch.no_grad():

    for t in reversed(range(1000)):

        timesteps = torch.full(
            (8,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        predicted_noise = model(
            x,
            timesteps,
            labels
        ).sample

        alpha = diffusion.alphas[t]
        alpha_bar = diffusion.alpha_bars[t]
        beta = diffusion.betas[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
        )

        if t > 0:
            x = x + torch.sqrt(beta) * torch.randn_like(x)

        x = torch.clamp(x, -1, 1)

        if t in snapshot_timesteps:
            snapshots[t] = ((x + 1) / 2).clamp(0, 1).cpu()

fig, axes = plt.subplots(6, 8, figsize=(12, 10))

for row, t in enumerate(snapshot_timesteps):

    for col in range(8):

        axes[row, col].imshow(
            snapshots[t][col, 0],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(
        f"t = {t}",
        rotation=0,
        labelpad=35
    )

plt.suptitle("DDPM: Reverse Denoising Process")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "ddpm_denoising_snapshots.png"),
    dpi=200
)

plt.close()

print("Denoising snapshots saved.")
print("All DDPM visualizations completed.")
