
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from ml.dataset import (
    create_dataloaders,
    CLASS_NAMES,
)
from ml.model import get_model


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "wheat_resnet18_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "models" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. LOAD SAVED MODEL
# --------------------------------------------------

def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = get_model(
        device=DEVICE,
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    return model


# --------------------------------------------------
# 3. TEST EVALUATION
# --------------------------------------------------

def evaluate():

    print("\n========== KHETMITRA MODEL EVALUATION ==========")

    print("Device:", DEVICE)

    model = load_model()

    # Load the untouched test split
    _, _, test_loader = create_dataloaders(
        batch_size=16
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):

                outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    # Calculate metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    print("\n========== OVERALL RESULTS ==========")

    print(f"Test Accuracy: {accuracy:.4f}")

    print(f"Test Macro F1: {macro_f1:.4f}")

    # Per-class metrics
    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display.plot(
        ax=ax,
        cmap="Greens",
        values_format="d",
    )

    plt.title("KhetMitra AI - Wheat Disease Confusion Matrix")

    plt.xticks(rotation=20)

    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nConfusion matrix saved:", confusion_path)

    print("\n========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    evaluate()
