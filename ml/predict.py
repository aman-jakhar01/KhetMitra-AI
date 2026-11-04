
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from ml.model import WheatDiseaseModel


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "wheat_resnet18_best.pth"
)

CLASS_NAMES = [
    "healthy",
    "nitrogen_deficiency",
    "leaf_rust",
]

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "nitrogen_deficiency": "Nitrogen Deficiency",
    "leaf_rust": "Leaf Rust",
}


# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

inference_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

])


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # Verify class names
    checkpoint_classes = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    if checkpoint_classes != CLASS_NAMES:

        raise ValueError(
            f"Unexpected model classes: {checkpoint_classes}"
        )

    # Create model architecture
    model = WheatDiseaseModel(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    print(
        f"Crop Doctor model loaded on {device}"
    )

    return model, device


# ==========================================
# 4. IMAGE LOADING
# ==========================================

def load_image(image):

    """
    Supports:
    1. Image file paths
    2. PIL images
    3. Streamlit UploadedFile objects
    """

    if isinstance(image, Image.Image):

        pil_image = image

    elif isinstance(image, (str, Path)):

        pil_image = Image.open(image)

    else:

        # Streamlit UploadedFile is file-like
        image.seek(0)

        pil_image = Image.open(image)

    return pil_image.convert("RGB")


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_image(image):

    # Load cached model
    model, device = load_model()

    # Read and preprocess image
    pil_image = load_image(image)

    image_tensor = inference_transform(
        pil_image
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move tensor to GPU or CPU
    image_tensor = image_tensor.to(device)

    # Model inference
    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    # Find highest probability class
    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ].item() * 100

    # Calculate all class probabilities
    all_probabilities = {}

    for index, class_name in enumerate(CLASS_NAMES):

        display_name = DISPLAY_NAMES[
            class_name
        ]

        probability = probabilities[
            index
        ].item() * 100

        all_probabilities[
            display_name
        ] = round(probability, 2)

    # Return results
    return {

        "predicted_class": predicted_class,

        "display_name": DISPLAY_NAMES[
            predicted_class
        ],

        "confidence": round(
            confidence,
            2
        ),

        "probabilities": all_probabilities,

        "device": str(device),

    }
