
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load the already-downloaded training dataset.
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)

# Display the first 16 images in a 4 x 4 grid.
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for index, ax in enumerate(axes.flat):
    image, label = dataset[index]

    # Convert PyTorch format [C, H, W] to image format [H, W, C].
    image = image.permute(1, 2, 0).numpy()

    ax.imshow(image)
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.tight_layout()

# Save a copy for your research records.
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/cifar10_samples.png", dpi=150)
print("Saved image grid to outputs/cifar10_samples.png")

plt.show()
