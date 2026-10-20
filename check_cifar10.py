
import torch 
import torchvision
from torchvision import datasets, transforms

# Convert each image into a PyTorch tensor.
transform = transforms.ToTensor()

# Download and load the training and test datasets.
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label ID:", label)
print("Pixel range:", image.min().item(), "to", image.max().item())
print("Class names:", train_dataset.classes)
