from pathlib import Path
import csv
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from ml.model import WheatDiseaseModel

# Run this script from the project root: D:\agri
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "wheat_resnet18_best.pth"

# Dataset folders based on the project structure shared.
NDEF_ROOT = ROOT / "data" / "crop_disease" / "wheat" / "Ndeficient" / "Ndeficient"
RUST_ROOT = ROOT / "data" / "crop_disease" / "wheat" / "WheatLeafRust" / "WheatLeafRust"

CLASS_NAMES = ["healthy", "nitrogen_deficiency", "leaf_rust"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class EvaluationImageDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, source_split = self.records[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = transform(image)
        return tensor, label, str(path), source_split

def collect_images(folder, label, source_split, records):
    if not folder.exists():
        print(f"WARNING: Folder not found, skipping: {folder}")
        return
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    records.extend((p, label, source_split) for p in files)
    print(f"{source_split}: {len(files)} images -> {label}")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    records = []

    # Nitrogen-deficiency validation split (not a test split).
    collect_images(NDEF_ROOT / "val" / "control", 0,
                   "nitrogen_validation_control", records)
    collect_images(NDEF_ROOT / "val" / "deficiency", 1,
                   "nitrogen_validation_deficiency", records)

    # Wheat leaf-rust independent test split.
    collect_images(RUST_ROOT / "test" / "control", 0,
                   "rust_test_control", records)
    collect_images(RUST_ROOT / "test" / "diseased", 2,
                   "rust_test_diseased", records)

    if not records:
        raise RuntimeError(
            "No images found. Check the dataset paths at the top of this script."
        )

    # Validate checkpoint class order where metadata is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    checkpoint_classes = checkpoint.get("class_names", CLASS_NAMES)
    if checkpoint_classes != CLASS_NAMES:
        raise ValueError(
            f"Checkpoint classes {checkpoint_classes} do not match {CLASS_NAMES}"
        )

    model = WheatDiseaseModel(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        EvaluationImageDataset(records),
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Windows-friendly
    )

    y_true, y_pred, rows = [], [], []
    with torch.inference_mode():
        for images, labels, paths, splits in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().tolist()
            confidence = probabilities.max(dim=1).values.cpu().tolist()

            for true_label, pred_label, conf, path, split in zip(
                labels.tolist(), predictions, confidence, paths, splits
            ):
                y_true.append(true_label)
                y_pred.append(pred_label)
                rows.append({
                    "image_path": path,
                    "source_split": split,
                    "actual_class": CLASS_NAMES[true_label],
                    "predicted_class": CLASS_NAMES[pred_label],
                    "confidence_percent": round(conf * 100, 2),
                    "correct": true_label == pred_label,
                })

    print("\n===== OVERALL EVALUATION =====")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Device: {device}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    # Save detailed predictions.
    output_dir = ROOT / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "crop_doctor_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save a simple confusion-matrix image.
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm)
    ax.set(
        xticks=range(3), yticks=range(3),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted class", ylabel="Actual class",
        title="Crop Doctor Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    cm_path = output_dir / "crop_doctor_confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    # Save metrics and split caveat alongside results.
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "images_evaluated": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "evaluation_note": (
            "This is a combined holdout evaluation: the leaf-rust data uses "
            "its test split, while nitrogen-deficiency data uses its validation "
            "split because no nitrogen test split was identified. Do not describe "
            "the combined score as a fully independent three-class test score."
        ),
    }
    json_path = output_dir / "crop_doctor_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved results to:")
    print(csv_path)
    print(cm_path)
    print(json_path)
    print("\nNOTE: The combined score mixes a rust test split with a nitrogen "
          "validation split. Report that limitation clearly.")

if __name__ == "__main__":
    main()
