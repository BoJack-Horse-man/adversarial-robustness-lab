import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18(num_classes=10):
    model = resnet18(weights=None)

    # Adapt ResNet-18 for 32x32 CIFAR-10 images.
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    # Remove ImageNet's initial downsampling.
    model.maxpool = nn.Identity()

    # Change the classifier from 1000 classes to 10.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model