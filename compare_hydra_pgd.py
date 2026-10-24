import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Use the separate HYDRA reference repository without modifying it.
sys.path.insert(0, str(Path(r"..\hydra-reference").resolve()))

import models
from utils.model import get_layers, prepare_model
from utils.adv import pgd_whitebox


SEED = 2026
EPSILON = 8 / 255
STEP_SIZE = 2 / 255
NUM_STEPS = 10
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = Path(
    r"..\hydra-pilot-results\cifar10-subnet-pretrain\pretrain"
    r"\latest_exp\checkpoint\model_best.pth.tar"
)
DATA_DIR = Path(r"..\hydra-pilot-data\CIFAR10")


def hydra_pgd_from_start(model, x, y, x_start):
    """Reproduce HYDRA's update steps from a supplied starting image."""
    x_original = x.detach()
    x_adv = x_start.detach().requires_grad_(True)

    for _ in range(NUM_STEPS):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = (x_adv.detach() + STEP_SIZE * grad.sign()).detach()

        delta = torch.clamp(
            x_adv - x_original, min=-EPSILON, max=EPSILON
        )
        x_adv = (x_original + delta).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()


def corrected_pgd(model, x, y, x_start):
    """PGD from the same start, with valid initialization and projection."""
    x_original = x.detach()
    x_adv = torch.clamp(x_start.detach(), 0.0, 1.0)

    # Project the supplied start onto the valid epsilon ball.
    delta = torch.clamp(
        x_adv - x_original, min=-EPSILON, max=EPSILON
    )
    x_adv = torch.clamp(x_original + delta, 0.0, 1.0).detach()

    for _ in range(NUM_STEPS):
        x_adv = x_adv.detach().requires_grad_(True)

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + STEP_SIZE * grad.sign()
        delta = torch.clamp(
            x_adv - x_original, min=-EPSILON, max=EPSILON
        )
        x_adv = torch.clamp(x_original + delta, 0.0, 1.0).detach()

    return x_adv

def accuracy(model, x, y):
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return (predictions == y).float().mean().item() * 100


def audit(name, x_adv, x, y, model):
    delta = (x_adv - x).abs()
    print(f"\n{name}")
    print(f"  Adversarial accuracy: {accuracy(model, x_adv, y):.2f}%")
    print(f"  Maximum perturbation: {delta.max().item():.8f}")
    print(f"  Epsilon limit:       {EPSILON:.8f}")
    print(f"  Within epsilon:       {delta.max().item() <= EPSILON + 1e-6}")
    print(
        "  Pixels in [0,1]:     "
        f"{x_adv.min().item() >= 0 and x_adv.max().item() <= 1}"
    )
    print(f"  Finite values:        {torch.isfinite(x_adv).all().item()}")


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT.resolve()}")

    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 data not found: {DATA_DIR.resolve()}")

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT.resolve())
    print("Dataset:", DATA_DIR.resolve())

    args = SimpleNamespace(
        layer_type="subnet",
        exp_mode="pretrain",
        k=1.0,
        freeze_bn=False,
        scores_init_type="kaiming_normal",
    )
    conv, linear = get_layers("subnet")
    model = models.cifar_model(
        conv, linear, "kaiming_normal", num_classes=10
    ).to(DEVICE)
    prepare_model(model, args)

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)

    print(f"\nAudit batch size: {len(y)}")
    print(f"Clean accuracy: {accuracy(model, x, y):.2f}%")

    # Use the same raw random start for both attacks.
    torch.manual_seed(SEED)
    noise = torch.empty(
        x.shape, dtype=x.dtype, device="cpu"
    ).uniform_(-EPSILON, EPSILON).to(DEVICE)
    raw_start = (x.detach() + noise).detach()

    print(
        "Raw start pixels in [0,1]:",
        raw_start.min().item() >= 0 and raw_start.max().item() <= 1,
    )

    original_adv = hydra_pgd_from_start(model, x, y, raw_start)
    corrected_adv = corrected_pgd(model, x, y, raw_start)

    audit("Original HYDRA PGD", original_adv.detach(), x, y, model)
    audit("Corrected PGD", corrected_adv.detach(), x, y, model)

    difference = (original_adv.detach() - corrected_adv.detach()).abs()
    print("\nDifference between adversarial images")
    print(f"  Maximum absolute difference: {difference.max().item():.10f}")
    print(f"  Mean absolute difference:    {difference.mean().item():.10f}")
    print(f"  Exactly identical:           {torch.equal(original_adv, corrected_adv)}")

    print("\nAudit complete. This is a diagnostic batch, not a full evaluation.")


if __name__ == "__main__":
    main()
