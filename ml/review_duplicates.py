
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw

from ml.dataset import WheatCropDataset, CLASS_NAMES


OUTPUT_DIR = Path("data/crop_disease/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

# Load and hash images from all splits
for split in ["train", "val", "test"]:

    dataset = WheatCropDataset(split=split)

    for image_path, label in dataset.samples:

        with Image.open(image_path) as img:
            phash = imagehash.phash(img.convert("RGB"))

        records.append({
            "path": image_path,
            "split": split,
            "label": CLASS_NAMES[label],
            "hash": phash,
        })


# Compare images belonging to different splits
candidates = []

for a, b in combinations(records, 2):

    if a["split"] == b["split"]:
        continue

    distance = a["hash"] - b["hash"]

    if distance <= 4:

        candidates.append({
            "a": a,
            "b": b,
            "distance": distance,
            "label_mismatch": a["label"] != b["label"],
        })


# Prioritize different-label matches and smallest distances
candidates.sort(
    key=lambda x: (
        not x["label_mismatch"],
        x["distance"],
    )
)

# Save a report
report_path = OUTPUT_DIR / "duplicate_candidates.txt"

with open(report_path, "w", encoding="utf-8") as f:

    for item in candidates:

        f.write(
            f"Distance: {item['distance']}\n"
            f"Label mismatch: {item['label_mismatch']}\n"
            f"A: {item['a']['path']} "
            f"[{item['a']['split']}, {item['a']['label']}]\n"
            f"B: {item['b']['path']} "
            f"[{item['b']['split']}, {item['b']['label']}]\n\n"
        )


# Create a contact sheet for visual inspection
selected = candidates[:24]

thumb_width = 240
thumb_height = 170
row_height = 235

sheet = Image.new(
    "RGB",
    (thumb_width * 2 + 40, row_height * len(selected)),
    "white",
)

draw = ImageDraw.Draw(sheet)

for index, item in enumerate(selected):

    y = index * row_height

    for position, record in enumerate([item["a"], item["b"]]):

        x = 10 + position * (thumb_width + 10)

        with Image.open(record["path"]) as img:
            thumbnail = img.convert("RGB")
            thumbnail.thumbnail((thumb_width, thumb_height))

        sheet.paste(thumbnail, (x, y + 30))

        draw.text(
            (x, y + 5),
            f"{record['split']} | {record['label']}",
            fill="black",
        )

    status = (
        f"Distance: {item['distance']} | "
        f"Different labels: {item['label_mismatch']}"
    )

    draw.text(
        (10, y + 205),
        status,
        fill="red" if item["label_mismatch"] else "black",
    )


sheet.save(OUTPUT_DIR / "duplicate_review.png")

print("Total candidates:", len(candidates))
print("Different-label candidates:", sum(
    item["label_mismatch"] for item in candidates
))
print("Review image:", OUTPUT_DIR / "duplicate_review.png")
print("Full report:", report_path)
