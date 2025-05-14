import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel
from tqdm import tqdm
import os

def optimize_discretise(integrator="midpoint", step_size=0.05, device="cuda"):
    os.makedirs("outputs", exist_ok=True)

    model_cfg = {
        "in_channels": 1, "model_channels": 32, "out_channels": 1,
        "num_res_blocks": 4, "attention_resolutions": [2], "dropout": 0.0,
        "channel_mult": [2, 2, 2], "conv_resample": False, "dims": 2,
        "num_classes": None, "use_checkpoint": False, "num_heads": 1,
        "num_head_channels": -1, "num_heads_upsample": -1,
        "use_scale_shift_norm": True, "resblock_updown": False,
        "use_new_attention_order": True, "with_fourier_features": False,
        "max_period": 2.0
    }

    vf = UNetModel(**model_cfg)
    vf.load_state_dict(torch.load("flow_matching_mnist.pt", map_location=device))
    vf = vf.to(device).eval()

    class WrappedModel(ModelWrapper):
        def forward(self, x, t, **extras):
            return self.model(x, t)

    wrapped_vf = WrappedModel(vf)
    dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
    x = dataset[1][0].unsqueeze(0).to(device)

    A = torch.randn(256, 28 * 28).to(device) / 256
    y = torch.matmul(x.reshape(1, -1), A.T)
    y_noise = y + 0.01 * torch.mean(torch.abs(y)) * torch.randn_like(y)

    solver = ODESolver(velocity_model=wrapped_vf)
    z0 = torch.nn.Parameter(torch.randn(1, 1, 28, 28).to(device))
    optimizer = torch.optim.Adam([z0], lr=1e-1)

    for i in tqdm(range(200)):
        optimizer.zero_grad()
        batch_t = torch.linspace(0., 1., 10).to(device)
        x_sample = solver.sample(batch_t, z0, method=integrator, step_size=step_size, enable_grad=True)
        loss = ((torch.matmul(x_sample.reshape(1, -1), A.T) - y_noise) ** 2).sum() + 0.01 * torch.mean(z0 ** 2)
        loss.backward()
        optimizer.step()

        if i % 4 == 0 and i > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(x[0, 0].cpu().numpy(), cmap="gray")
            ax1.set_title("GT")
            ax2.imshow(x_sample[0, 0].detach().cpu().numpy(), cmap="gray")
            ax2.set_title("Reconstructed")
            plt.savefig(f"outputs/img_rk_{integrator}_{i}.png")
            plt.close()
