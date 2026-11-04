
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

from ml.dataset import (
    WheatCropDataset,
    eval_transforms,
    CLASS_NAMES,
)
from ml.model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = PROJECT_ROOT / "models" / "wheat_resnet18_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "models" / "evaluation" / "errors"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    # Load model
    checkpoint = torch.load(
        CHECKPOINT,
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

    # Load test images
    dataset = WheatCropDataset(
        split="test",
        transform=eval_transforms,
    )

    errors = []

    with torch.no_grad():

        for index, (image, label) in enumerate(dataset):

            image = image.unsqueeze(0).to(DEVICE)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):

                output = model(image)

            probabilities = torch.softmax(
                output.float(),
                dim=1,
            )[0]

            prediction = probabilities.argmax().item()

            confidence = probabilities[prediction].item()

            if prediction != label:

                image_path = dataset.samples[index][0]

                errors.append({
                    "path": image_path,
                    "actual": CLASS_NAMES[label],
                    "predicted": CLASS_NAMES[prediction],
                    "confidence": confidence,
                })

    print("\n========== ERROR ANALYSIS ==========")

    print("Total incorrect predictions:", len(errors))

    for error in errors:

        print("\nImage:", error["path"])
        print("Actual:", error["actual"])
        print("Predicted:", error["predicted"])
        print(
            "Prediction confidence:",
            f"{error['confidence']:.2%}"
        )

    # Save misclassified images for visual inspection
    for index, error in enumerate(errors):

        with Image.open(error["path"]) as image:

            image = image.convert("RGB")

            image.save(
                OUTPUT_DIR / f"error_{index + 1}.jpg"
            )

    print("\nMisclassified images saved to:")
    print(OUTPUT_DIR)

    print("\nError analysis complete!")


if __name__ == "__main__":
    main()
