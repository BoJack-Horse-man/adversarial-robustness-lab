
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 100)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def fgsm_attack(model, images, labels, epsilon, criterion):
    """Create untargeted FGSM adversarial images."""
    images = images.detach().clone().requires_grad_(True)

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    perturbation = epsilon * images.grad.sign()
    adversarial_images = images.detach() + perturbation

    return torch.clamp(adversarial_images, 0.0, 1.0).detach()


def evaluate(model, loader, device, epsilon):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    model.eval()

    for images, labels in tqdm(loader, desc=f"epsilon={epsilon:.5f}"):
        images = images.to(device)
        labels = labels.to(device)

        if epsilon == 0:
            with torch.no_grad():
                outputs = model(images)
        else:
            adversarial_images = fgsm_attack(
                model, images, labels, epsilon, criterion
            )
            with torch.no_grad():
                outputs = model(adversarial_images)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = datasets.CIFAR100(
        root="./data_cifar100",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )

    model = SimpleCNN().to(device)
    weights_path = "checkpoints/simple_cnn_cifar100.pth"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Missing {weights_path}. Run train_cifar100_baseline.py first."
        )

    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    epsilons = [0.0, 2 / 255, 4 / 255, 8 / 255]
    results = []

    for epsilon in epsilons:
        accuracy = evaluate(model, loader, device, epsilon)
        results.append((epsilon, accuracy))

        label = "Clean" if epsilon == 0 else f"FGSM epsilon={epsilon:.5f}"
        print(f"{label}: accuracy={accuracy:.2f}%")

    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/cifar100_fgsm_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epsilon", "test_accuracy_percent"])
        writer.writerows(results)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
