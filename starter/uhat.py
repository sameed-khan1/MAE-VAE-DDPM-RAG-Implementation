
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_split_dir(root, kind, split):
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    for directory in sorted(root.rglob("*")):
        name = directory.name.lower()

        if directory.is_dir() and kind in name and split in name:
            if any(child.is_dir() for child in directory.iterdir()):
                return directory

    found = [
        str(directory)
        for directory in root.rglob("*")
        if directory.is_dir()
    ][:30]

    raise FileNotFoundError(
        f"Could not find a folder containing '{kind}' and '{split}' "
        f"under {root}. Folders found: {found}"
    )


def load_uhat_characters(
    root="data/uhat",
    split="train",
    class_names=None
):
    split_dir = _find_split_dir(root, "char", split)

    if class_names is None:
        class_names = sorted(
            child.name.lower()
            for child in split_dir.iterdir()
            if child.is_dir()
        )

    index = {
        name.lower(): i
        for i, name in enumerate(class_names)
    }

    images = []
    labels = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        if class_name not in index:
            continue

        for file_path in sorted(class_dir.rglob("*")):
            if file_path.suffix.lower() not in IMG_EXT:
                continue

            with Image.open(file_path) as image:
                image = image.convert("L").resize((28, 28))
                image_array = np.asarray(image, dtype=np.float32) / 255.0

            images.append(image_array)
            labels.append(index[class_name])

    if not images:
        raise RuntimeError(
            f"No character images found in {split_dir}. "
            "Check the extracted UHaT folder structure."
        )

    x = torch.from_numpy(np.stack(images)).unsqueeze(1)

    flip = x.mean(dim=(1, 2, 3)) > 0.5
    x[flip] = 1.0 - x[flip]

    y = torch.tensor(labels, dtype=torch.long)

    return x, y, list(class_names)
