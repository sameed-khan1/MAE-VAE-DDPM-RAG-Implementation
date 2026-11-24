
import os
import sys
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from starter.uhat import load_uhat_characters
from vae import VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = os.path.join(ROOT, "starter", "data", "uhat", "data")
CHECKPOINT = os.path.join(ROOT, "checkpoints", "vae_best.pth")
OUTPUT_DIR = os.path.join(ROOT, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_grid(images, path, title, nrow=8):

    images = images.detach().cpu().clamp(0, 1)

    count = len(images)
    ncol = nrow
    nrows = (count + ncol - 1) // ncol

    fig, axes = plt.subplots(
        nrows, ncol,
        figsize=(12, 1.6 * nrows),
        squeeze=False
    )

    for i, ax in enumerate(axes.flat):

        ax.axis("off")

        if i < count:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():

    model = VAE(latent_dim=16).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("VAE checkpoint loaded.")

    test_images, test_labels, class_names = load_uhat_characters(
        root=DATA_ROOT,
        split="test"
    )

    # 1. Original images and reconstructions

    images = test_images[:16].to(DEVICE)

    with torch.no_grad():
        reconstruction, _, _ = model(images)
        reconstruction = torch.sigmoid(reconstruction)

    fig, axes = plt.subplots(2, 16, figsize=(20, 4))

    for i in range(16):

        axes[0, i].imshow(images[i, 0].cpu(), cmap="gray")
        axes[0, i].set_title("Original", fontsize=7)

        axes[1, i].imshow(
            reconstruction[i, 0].cpu(),
            cmap="gray"
        )
        axes[1, i].set_title("Reconstructed", fontsize=7)

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "vae_reconstructions.png"),
        dpi=150
    )

    plt.close()

    print("Reconstructions saved.")

    # 2. Generate 64 new characters

    with torch.no_grad():
        generated = model.sample(64, DEVICE)

    save_grid(
        generated,
        os.path.join(OUTPUT_DIR, "vae_generated_samples.png"),
        "64 Characters Generated from Random Latent Vectors"
    )

    print("Generated samples saved.")

    # 3. Interpolation between two different characters

    first_index = None
    second_index = None

    for i in range(len(test_labels)):
        for j in range(i + 1, len(test_labels)):
            if test_labels[i] != test_labels[j]:
                first_index = i
                second_index = j
                break

        if first_index is not None:
            break

    if first_index is None:
        raise RuntimeError("Could not find two different character classes.")

    first_image = test_images[first_index:first_index + 1].to(DEVICE)
    second_image = test_images[second_index:second_index + 1].to(DEVICE)

    with torch.no_grad():

        mu1, _ = model.encode(first_image)
        mu2, _ = model.encode(second_image)

        alphas = torch.linspace(0, 1, 10, device=DEVICE).unsqueeze(1)

        interpolated_z = (
            (1 - alphas) * mu1
            + alphas * mu2
        )

        interpolated_images = torch.sigmoid(
            model.decode(interpolated_z)
        )

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))

    for i, ax in enumerate(axes):

        ax.imshow(
            interpolated_images[i, 0].cpu(),
            cmap="gray"
        )

        ax.set_title(f"{i + 1}", fontsize=9)
        ax.axis("off")

    plt.suptitle(
        f"Interpolation: {class_names[test_labels[first_index]]} "
        f"to {class_names[test_labels[second_index]]}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "vae_interpolation.png"),
        dpi=150
    )

    plt.close()

    print("Interpolation saved.")
    print("\nAll VAE visualizations completed.")


if __name__ == "__main__":
    main()
