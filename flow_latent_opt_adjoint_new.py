import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel
from tqdm import tqdm
import os

def optimize_adjoint(integrator="midpoint", step_size=0.025, device="cuda"):
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

    A = torch.randn(200, 28*28).to(device) / 200
    y = torch.matmul(x.reshape(1, -1), A.T)
    y_noise = y + 0.01 * torch.mean(torch.abs(y)) * torch.randn_like(y)

    solver = ODESolver(velocity_model=wrapped_vf)
    z0 = torch.nn.Parameter(torch.randn(1, 1, 28, 28).to(device))
    optimizer = torch.optim.Adam([z0], lr=1e-1)

    for i in tqdm(range(200)):
        optimizer.zero_grad()
        batch_t = torch.linspace(0., 1., 10).to(device)
        x_sample = solver.sample(
            time_grid=batch_t,
            x_init=z0,
            method=integrator,
            step_size=step_size,
            return_intermediates=False
        )
        x_sample.requires_grad_(True)

        loss = ((torch.matmul(x_sample.reshape(1, -1), A.T) - y_noise)**2).sum() + 0.01 * torch.mean(z0**2)
        lambda_ = torch.autograd.grad(loss, x_sample, create_graph=True)[0]

        # Reverse-time ODE for adjoint (RK4)
        def rk4_reverse(lambda_t, x_t, t):
            h = step_size
            def grad_fn(x):
                return wrapped_vf(x, t)

            k1 = torch.autograd.functional.jvp(grad_fn, (x_t,), (lambda_t,))[1]
            k2 = torch.autograd.functional.jvp(grad_fn, (x_t - 0.5*h*k1,), (lambda_t,))[1]
            k3 = torch.autograd.functional.jvp(grad_fn, (x_t - 0.5*h*k2,), (lambda_t,))[1]
            k4 = torch.autograd.functional.jvp(grad_fn, (x_t - h*k3,), (lambda_t,))[1]
            return lambda_t - (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        t = torch.tensor(1.0).to(device)
        xt = x_sample.detach()
        for _ in range(int(1/step_size)):
            lambda_ = rk4_reverse(lambda_, xt, t)
            xt = xt.detach()
            t -= step_size

        z0.grad = lambda_ + 0.01 * z0
        optimizer.step()

        if i % 4 == 0 and i > 0:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
            ax1.set_title("GT")
            ax2.imshow(x_sample[0,0].detach().cpu().numpy(), cmap="gray")
            ax2.set_title("Reconstructed")
            plt.savefig(f"outputs/img_adjoint_{integrator}_{i}.png")
            plt.close()
