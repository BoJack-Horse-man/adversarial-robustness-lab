import sys
sys.path.insert(0, r"..\hydra-reference")

import torch
import models
from types import SimpleNamespace
from utils.model import get_layers, prepare_model
from utils.adv import pgd_whitebox

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
).cuda()
prepare_model(model, args)
model.eval()

x = torch.rand(4, 3, 32, 32, device="cuda")
y = torch.randint(0, 10, (4,), device="cuda")
epsilon = 8 / 255

with torch.no_grad():
    x_adv = pgd_whitebox(
        model, x, y, torch.device("cuda"),
        epsilon, 3, 2 / 255, 0.0, 1.0, is_random=True
    )

delta = (x_adv - x).abs()
print("Output shape:", tuple(x_adv.shape))
print("Pixels within [0,1]:", x_adv.min().item() >= 0 and x_adv.max().item() <= 1)
print("Maximum perturbation:", delta.max().item())
print("Within epsilon:", delta.max().item() <= epsilon + 1e-6)
print("Finite outputs:", torch.isfinite(x_adv).all().item())