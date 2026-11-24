
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_grid(images, path, nrow=8, titles=None):
    images = images.detach().cpu().clamp(0, 1)

    n = images.shape[0]
    rows = (n + nrow - 1) // nrow

    fig, axes = plt.subplots(
        rows,
        nrow,
        figsize=(1.4 * nrow, 1.4 * rows),
        squeeze=False
    )

    for i, ax in enumerate(axes.flat):
        ax.axis("off")

        if i < n:
            image = images[i].permute(1, 2, 0).numpy()

            if image.shape[-1] == 1:
                ax.imshow(
                    image.squeeze(-1),
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )
            else:
                ax.imshow(image)

            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
