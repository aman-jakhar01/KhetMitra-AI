
import hashlib
from itertools import product

import imagehash
from PIL import Image

from ml.dataset import WheatCropDataset, CLASS_NAMES


# Load all three original splits
splits = {
    split: WheatCropDataset(split=split)
    for split in ["train", "val", "test"]
}


def calculate_sha256(image_path):
    """Calculate exact file hash."""

    hasher = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_phash(image_path):
    """Calculate visual perceptual hash."""

    with Image.open(image_path) as image:
        return imagehash.phash(image.convert("RGB"))


# Prepare image records
records = {}

for split, dataset in splits.items():

    records[split] = []

    for image_path, label in dataset.samples:

        records[split].append({
            "path": image_path,
            "label": CLASS_NAMES[label],
            "sha256": calculate_sha256(image_path),
            "phash": calculate_phash(image_path),
        })

    print(f"{split}: {len(records[split])} images audited")


# Compare different splits
split_pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]

exact_duplicates = []
near_duplicates = []

for split_a, split_b in split_pairs:

    for image_a, image_b in product(
        records[split_a],
        records[split_b]
    ):

        # Exact duplicate detection
        if image_a["sha256"] == image_b["sha256"]:

            exact_duplicates.append((
                split_a,
                split_b,
                image_a,
                image_b,
            ))

            continue

        # Perceptual duplicate detection
        distance = image_a["phash"] - image_b["phash"]

        if distance <= 4:

            near_duplicates.append((
                split_a,
                split_b,
                distance,
                image_a,
                image_b,
            ))


# Display results
print("\n========== DUPLICATE AUDIT ==========")

print("Exact duplicate pairs:", len(exact_duplicates))

print("Near-duplicate candidates:", len(near_duplicates))


if exact_duplicates:

    print("\nEXACT DUPLICATES:")

    for split_a, split_b, image_a, image_b in exact_duplicates[:20]:

        print(f"\n{split_a} <-> {split_b}")

        print("Image A:", image_a["path"])
        print("Image B:", image_b["path"])


if near_duplicates:

    print("\nNEAR-DUPLICATE CANDIDATES:")

    for split_a, split_b, distance, image_a, image_b in near_duplicates[:20]:

        print(f"\n{split_a} <-> {split_b}")

        print("Perceptual distance:", distance)

        print("Image A:", image_a["path"])
        print("Image B:", image_b["path"])


print("\n========== AUDIT COMPLETE ==========")
