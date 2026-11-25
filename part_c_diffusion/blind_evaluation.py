
import os
import sys
import csv
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DiffusionUNet
from diffusion import Diffusion


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "checkpoints", "ddpm_best.pth")
RESULTS = os.path.join(ROOT, "results")

os.makedirs(RESULTS, exist_ok=True)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = DiffusionUNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

diffusion = Diffusion(timesteps=1000, device=DEVICE)

all_images = []
all_labels = []

print("Generating one sample per class...")

for start in range(0, 40, 4):

    labels = torch.arange(
        start,
        min(start + 4, 40),
        device=DEVICE
    )

    samples = diffusion.sample(
        model,
        labels,
        image_size=28
    )

    all_images.append(samples.cpu())
    all_labels.extend(labels.cpu().tolist())

    print(f"Generated classes {start} to {min(start + 3, 39)}")


all_images = torch.cat(all_images, dim=0)

# Shuffle samples and their corresponding labels together
permutation = torch.randperm(40)

all_images = all_images[permutation]
all_labels = [all_labels[i] for i in permutation.tolist()]

# Save blind evaluation sheet
fig, axes = plt.subplots(5, 8, figsize=(14, 10))

for i, ax in enumerate(axes.flat):

    image = ((all_images[i, 0] + 1) / 2).clamp(0, 1)

    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Sample {i + 1:02d}")
    ax.axis("off")

plt.suptitle("DDPM Blind Character Recognition Evaluation")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS, "ddpm_blind_evaluation.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# Save answer key separately
with open(
    os.path.join(RESULTS, "ddpm_blind_answer_key.csv"),
    "w",
    newline=""
) as file:

    writer = csv.writer(file)
    writer.writerow(["sample_id", "actual_class"])

    for i, label in enumerate(all_labels):
        writer.writerow([i + 1, class_names[label]])

# Create blank human evaluation file
with open(
    os.path.join(RESULTS, "human_eval.csv"),
    "w",
    newline=""
) as file:

    writer = csv.writer(file)
    writer.writerow([
        "evaluator",
        "sample_id",
        "predicted_class"
    ])

    for i in range(1, 41):
        writer.writerow(["", i, ""])


print("\nBlind evaluation sheet saved.")
print("Answer key saved separately.")
print("Human evaluation CSV created.")
