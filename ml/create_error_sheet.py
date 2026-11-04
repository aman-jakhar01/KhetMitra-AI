
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ml.dataset import WheatCropDataset, eval_transforms, CLASS_NAMES
from ml.model import get_model


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "models" / "wheat_resnet18_best.pth"

OUTPUT = ROOT / "models" / "evaluation" / "error_contact_sheet.jpg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model = get_model(
        device=DEVICE,
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = WheatCropDataset(
        split="test",
        transform=eval_transforms,
    )

    errors = []

    with torch.no_grad():

        for index, (image, label) in enumerate(dataset):

            image_batch = image.unsqueeze(0).to(DEVICE)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):
                output = model(image_batch)

            probabilities = torch.softmax(
                output.float(), dim=1
            )[0]

            prediction = probabilities.argmax().item()

            if prediction != label:

                path = dataset.samples[index][0]

                errors.append({
                    "path": path,
                    "actual": CLASS_NAMES[label],
                    "predicted": CLASS_NAMES[prediction],
                    "confidence": probabilities[prediction].item(),
                    "index": index,
                })

    columns = 3
    cell_width = 320
    cell_height = 285

    rows = (len(errors) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for i, error in enumerate(errors):

        x = (i % columns) * cell_width
        y = (i // columns) * cell_height

        with Image.open(error["path"]) as original:

            image = original.convert("RGB")
            image.thumbnail((280, 190))

        sheet.paste(image, (x + 15, y + 10))

        source = error["path"].parts[-5]

        draw.text(
            (x + 10, y + 210),
            f"Actual: {error['actual']}",
            fill="black",
        )

        draw.text(
            (x + 10, y + 230),
            f"Predicted: {error['predicted']}",
            fill="red",
        )

        draw.text(
            (x + 10, y + 250),
            f"Confidence: {error['confidence']:.1%}",
            fill="black",
        )

    sheet.save(OUTPUT, quality=95)

    print("Incorrect predictions:", len(errors))
    print("Contact sheet:", OUTPUT)


if __name__ == "__main__":
    main()
