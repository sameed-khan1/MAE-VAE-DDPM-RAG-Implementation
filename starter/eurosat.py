
import numpy as np
import torch
from datasets import load_dataset

EUROSAT_REPO = "blanchon/EuroSAT_RGB"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

PATCH = 8


def load_eurosat_split(split):
    ds = load_dataset(EUROSAT_REPO, split=split)

    imgs = np.stack([
        np.asarray(im.convert("RGB"), dtype=np.uint8)
        for im in ds["image"]
    ])

    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(ds["label"], dtype=torch.long)

    return images, labels


def few_label_indices(labels, per_class=16, seed=42):
    rng = np.random.default_rng(seed)
    idx = []

    for c in range(len(CLASS_NAMES)):
        members = np.flatnonzero(labels.numpy() == c)
        idx.extend(
            rng.choice(members, size=per_class, replace=False).tolist()
        )

    return torch.tensor(sorted(idx), dtype=torch.long)


def patchify(images, p=PATCH):
    n, c, h, w = images.shape

    if h % p != 0 or w % p != 0:
        raise ValueError("Image dimensions must be divisible by patch size.")

    x = images.reshape(n, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1)

    return x.reshape(n, (h // p) * (w // p), p * p * c)


def unpatchify(patches, p=PATCH, c=3):
    n, length, _ = patches.shape

    h = w = int(length ** 0.5)

    if h * w != length:
        raise ValueError("Number of patches must form a square grid.")

    x = patches.reshape(n, h, w, p, p, c)
    x = x.permute(0, 5, 1, 3, 2, 4)

    return x.reshape(n, c, h * p, w * p)
