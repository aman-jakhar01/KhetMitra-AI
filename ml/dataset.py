
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# 1. PROJECT PATHS AND CLASS CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "crop_disease"
    / "wheat"
)

CLASS_NAMES = [
    "healthy",
    "nitrogen_deficiency",
    "leaf_rust",
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# --------------------------------------------------
# 2. ORIGINAL DATASET LABEL MAPPING
# --------------------------------------------------

DATASET_CONFIG = {
    "Ndeficient": {
        "folder": "Ndeficient",
        "labels": {
            "control": "healthy",
            "deficiency": "nitrogen_deficiency",
        },
    },

    "WheatLeafRust": {
        "folder": "WheatLeafRust",
        "labels": {
            "control": "healthy",
            "diseased": "leaf_rust",
        },
    },
}


# --------------------------------------------------
# 3. IMAGE TRANSFORMATIONS
# --------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# --------------------------------------------------
# 4. CUSTOM PYTORCH DATASET
# --------------------------------------------------

class WheatCropDataset(Dataset):

    def __init__(self, split="train", transform=None):

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "Split must be train, val, or test"
            )

        self.split = split
        self.transform = transform
        self.samples = []

        self._load_samples()

        if not self.samples:
            raise RuntimeError(
                f"No images found for split: {split}"
            )

    def _load_samples(self):

        for dataset_name, config in DATASET_CONFIG.items():

            dataset_root = (
                DATA_ROOT
                / config["folder"]
                / config["folder"]
                / self.split
            )

            if not dataset_root.exists():
                raise FileNotFoundError(
                    f"Dataset folder not found: {dataset_root}"
                )

            for original_label, mapped_label in config["labels"].items():

                class_folder = dataset_root / original_label

                if not class_folder.exists():
                    raise FileNotFoundError(
                        f"Class folder missing: {class_folder}"
                    )

                class_index = CLASS_TO_IDX[mapped_label]

                for image_path in class_folder.rglob("*"):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    ):

                        self.samples.append(
                            (image_path, class_index)
                        )

        self.samples.sort(
            key=lambda item: str(item[0]).lower()
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            ) from error

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# 5. CREATE PYTORCH DATALOADERS
# --------------------------------------------------

def create_dataloaders(batch_size=16):

    train_dataset = WheatCropDataset(
        split="train",
        transform=train_transforms,
    )

    val_dataset = WheatCropDataset(
        split="val",
        transform=eval_transforms,
    )

    test_dataset = WheatCropDataset(
        split="test",
        transform=eval_transforms,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# 6. DATASET SUMMARY
# --------------------------------------------------

def print_dataset_summary():

    print("\n========== WHEAT DATASET SUMMARY ==========\n")

    for split in ["train", "val", "test"]:

        dataset = WheatCropDataset(split=split)

        counts = {
            class_name: 0
            for class_name in CLASS_NAMES
        }

        for _, label in dataset.samples:
            counts[CLASS_NAMES[label]] += 1

        print(f"{split.upper()}")

        for class_name, count in counts.items():
            print(f"  {class_name}: {count}")

        print(f"  Total: {len(dataset)}\n")

    print("Class mapping:", CLASS_TO_IDX)
    print("===========================================\n")
