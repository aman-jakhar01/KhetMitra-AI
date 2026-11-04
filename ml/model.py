
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class WheatDiseaseModel(nn.Module):

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        freeze_backbone=True,
    ):

        super().__init__()

        # Load ResNet18 with pretrained ImageNet weights
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(weights=weights)

        # Freeze all pretrained parameters initially
        if freeze_backbone:

            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            # Allow the final residual block to learn
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True

        # Get the number of input features
        input_features = self.backbone.fc.in_features

        # Replace ImageNet's 1000-class output layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(input_features, num_classes),
        )

    def forward(self, x):

        return self.backbone(x)


def get_model(device=None, pretrained=True):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = WheatDiseaseModel(
        num_classes=3,
        pretrained=pretrained,
        freeze_backbone=True,
    )

    model = model.to(device)

    return model
