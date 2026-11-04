
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st

from ml.model import WheatDiseaseModel
from config import DISEASE_MODEL_PATH


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

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

    model_path = Path(DISEASE_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Crop Doctor checkpoint not found: {model_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    checkpoint_classes = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    if checkpoint_classes != CLASS_NAMES:
        raise ValueError(
            f"Unexpected model classes: {checkpoint_classes}"
        )

    model = WheatDiseaseModel(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

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
        image.seek(0)
        pil_image = Image.open(image)

    return pil_image.convert("RGB")


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_image(image):

    model, device = load_model()

    pil_image = load_image(image)

    image_tensor = inference_transform(pil_image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = probabilities[
        predicted_index
    ].item() * 100

    all_probabilities = {}

    for index, class_name in enumerate(CLASS_NAMES):

        display_name = DISPLAY_NAMES[class_name]

        probability = probabilities[index].item() * 100

        all_probabilities[display_name] = round(
            probability,
            2
        )

    return {
        "predicted_class": predicted_class,
        "display_name": DISPLAY_NAMES[predicted_class],
        "confidence": round(confidence, 2),
        "probabilities": all_probabilities,
        "device": str(device),
    }


# ==========================================
# 6. MODEL INFORMATION
# ==========================================

def get_model_info():

    model, device = load_model()

    return {
        "model": "ResNet18",
        "classes": CLASS_NAMES,
        "device": str(device),
        "loaded": model is not None,
    }
