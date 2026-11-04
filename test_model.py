
import torch

from ml.model import get_model
from ml.dataset import CLASS_NAMES

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load model
model = get_model(device=device, pretrained=True)

# Evaluation mode
model.eval()

# Create a dummy batch
dummy_images = torch.randn(
    4, 3, 224, 224
).to(device)

# Forward pass
with torch.no_grad():

    outputs = model(dummy_images)

print("\n========== MODEL TEST ==========")

print("Output shape:", outputs.shape)

print("Expected classes:", len(CLASS_NAMES))

print("Class names:", CLASS_NAMES)

print("Model device:", next(model.parameters()).device)

print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

print("Total parameters:",
      sum(p.numel() for p in model.parameters()))

assert outputs.shape == (4, 3)

print("\nResNet18 model working successfully!")
