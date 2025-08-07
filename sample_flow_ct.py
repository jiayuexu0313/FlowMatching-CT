import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import yaml 

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from models.unet import UNetModel
from ct_dataset import CTDataset

device = "cuda"
batch_size = 6

with open("flow_matching_ct.yaml", "r") as file:
    model_cfg = yaml.safe_load(file)


vf = UNetModel(**model_cfg).to(device)
vf.load_state_dict(torch.load("flow_matching_ct.pt"))

vf.eval()

print("Number of params: ", sum([p.numel() for p in vf.parameters()]))

# do a unconditional sampling and save the image sample_epoch_{epoch}.png
with torch.no_grad():
    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return self.model(x, t)
    wrapped = Wrapped(vf)
    solver = ODESolver(velocity_model=wrapped)

    times = torch.linspace(0, 1, 100, device=device)
    x_init = torch.randn((6, 1, 128, 128), device=device)
    sol = solver.sample(time_grid=times, x_init=x_init, method="midpoint", step_size=times[1] - times[0])
    sample_imgs = sol.clamp(0, 1).cpu().numpy()

    print(sample_imgs.shape)

    fig, axes = plt.subplots(2, batch_size // 2, figsize=(12,6))

    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(sample_imgs[idx][0], cmap="gray")

        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_epoch_0.png")

    plt.show()
