
import torch

from ml.dataset import (
    create_dataloaders,
    print_dataset_summary,
    CLASS_NAMES,
)

# Print class counts
print_dataset_summary()

# Create loaders
train_loader, val_loader, test_loader = (
    create_dataloaders(batch_size=16)
)

# Load one training batch
images, labels = next(iter(train_loader))

print("\n========== BATCH TEST ==========")

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

print("Labels:", labels.tolist())

print("Class names:", [
    CLASS_NAMES[label]
    for label in labels.tolist()
])

print("GPU available:", torch.cuda.is_available())

print("Dataset loader working!")
