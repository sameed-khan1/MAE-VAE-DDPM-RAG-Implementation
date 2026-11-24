
import os
import sys
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from starter.eurosat import load_eurosat_split, patchify, unpatchify
from mae import MaskedAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_mae.pth")
OUTPUT = os.path.join(PROJECT_ROOT, "results", "mae_reconstructions.png")


def main():

    model = MaskedAutoencoder().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images, labels = load_eurosat_split("test")

    images = images[:8].to(DEVICE)
    patches = patchify(images)

    with torch.no_grad():
        predictions, mask = model(patches)

    masked_input = patches.clone()
    masked_input[mask.bool()] = 0.5

    reconstructed = patches.clone()
    reconstructed[mask.bool()] = predictions[mask.bool()]

    masked_images = unpatchify(masked_input)
    reconstructed_images = unpatchify(reconstructed)

    fig, axes = plt.subplots(3, 8, figsize=(16, 7))

    for i in range(8):

        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
        axes[0, i].set_title("Original")

        axes[1, i].imshow(masked_images[i].cpu().permute(1, 2, 0))
        axes[1, i].set_title("Masked")

        axes[2, i].imshow(
            reconstructed_images[i].cpu().permute(1, 2, 0).clamp(0, 1)
        )
        axes[2, i].set_title("Reconstructed")

        for row in range(3):
            axes[row, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    plt.close()

    print("Reconstruction saved:", OUTPUT)


if __name__ == "__main__":
    main()
