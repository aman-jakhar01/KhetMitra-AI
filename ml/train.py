
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

from ml.dataset import (
    create_dataloaders,
    WheatCropDataset,
    CLASS_NAMES,
)
from ml.model import get_model


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "wheat_resnet18_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# --------------------------------------------------
# 2. CLASS WEIGHTS
# --------------------------------------------------

def calculate_class_weights():

    dataset = WheatCropDataset(split="train")

    counts = torch.bincount(
        torch.tensor(
            [label for _, label in dataset.samples]
        ),
        minlength=len(CLASS_NAMES),
    ).float()

    weights = len(dataset) / (
        len(CLASS_NAMES) * counts
    )

    print("\nClass counts:")

    for name, count, weight in zip(
        CLASS_NAMES, counts, weights
    ):
        print(
            f"{name}: {int(count)} images | "
            f"Weight: {weight.item():.3f}"
        )

    return weights.to(DEVICE)


# --------------------------------------------------
# 3. TRAINING FUNCTION
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):

    model.train()

    # Keep frozen backbone BatchNorm statistics fixed
    model.backbone.bn1.eval()
    model.backbone.layer1.eval()
    model.backbone.layer2.eval()
    model.backbone.layer3.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

    epoch_loss = running_loss / len(loader.dataset)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, epoch_f1


# --------------------------------------------------
# 4. VALIDATION FUNCTION
# --------------------------------------------------

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP,
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    val_loss = running_loss / len(loader.dataset)

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return val_loss, val_accuracy, val_f1


# --------------------------------------------------
# 5. MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    print("\n========== KHETMITRA CROP DOCTOR ==========")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Mixed precision:", USE_AMP)

    # Load datasets
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    # Initialize model
    model = get_model(
        device=DEVICE,
        pretrained=True,
    )

    # Calculate class weights
    class_weights = calculate_class_weights()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Optimize only trainable parameters
    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(EPOCHS):

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | "
            f"Train Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        print(f"Learning Rate: {current_lr:.7f}")

        # Save best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_classes": len(CLASS_NAMES),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                },
                CHECKPOINT_PATH,
            )

            print("Best model checkpoint saved!")

        else:

            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    print("\n========== TRAINING COMPLETE ==========")

    print("Best validation loss:", best_val_loss)

    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
