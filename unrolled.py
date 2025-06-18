import os
import argparse
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import random

from deepinv.physics import Tomography
from ct_dataset import CTDataset
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel

def compute_mse(x_rec: torch.Tensor, x_gt: torch.Tensor) -> float:
    return torch.mean((x_rec - x_gt)**2).item()

def compute_psnr(x_rec: torch.Tensor, x_gt: torch.Tensor) -> float:
    mse = compute_mse(x_rec, x_gt)
    return 20 * np.log10(1.0 / np.sqrt(mse))

def optimise_z(physics, y, x_gt, sampler_steps, iter_max, device, weight_path="flow_matching_ct.pt", cfg_path="flow_matching_ct.yaml"):
    """First discretize (Midpoint) and then optimize z, and return x_rec, losses"""
    with open(cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    vf = UNetModel(**model_cfg).to(device).eval()
    vf.load_state_dict(torch.load(weight_path, map_location=device))
    
    class WrappedUNet(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, t, **kwargs):
            return self.model(x, t)
    wrapped = WrappedUNet(vf)

    solver = ODESolver(velocity_model=wrapped)


    z = torch.randn((1,1,128,128), device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=1e-1)
    losses = []
    psnrs  = [] # add psnrs over iterations

    step_size = 1.0 / sampler_steps
    for i in range(iter_max):
        opt.zero_grad()
        t_grid = torch.linspace(0, 1, sampler_steps, device=device)
        x_hat = solver.sample(
            time_grid=t_grid,
            x_init=z,
            method="midpoint",
            step_size=step_size,
            enable_grad=True
        )
        x_hat.requires_grad_(True)

        y_pred = (physics(x_hat).squeeze(0).squeeze(0).transpose(0,1))
        loss = ((y_pred - y)**2).sum() + 0.01 * (z**2).mean()
        losses.append(loss.item())

        loss.backward()
        opt.step()

        # record PSNR at this iteration
        with torch.no_grad():
            x_clamped = x_hat.clamp(0,1).detach()
            psnrs.append(compute_psnr(x_clamped, x_gt))

    x_rec = x_hat.clamp(0,1).detach()
    return x_rec, losses, psnrs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--angles",       type=int,   required=True,
                   help="CT projection angle")
    p.add_argument("--noise",        type=float, default=0.0,
                   help="relative noise ratio")
    p.add_argument("--sampler_steps",type=int,   default=6,
                   help="ODE sampling steps")
    p.add_argument("--iter_max",     type=int,   default=200,
                   help="the number of iterations")
    p.add_argument("--device",       choices=["cpu","cuda"],
                   default="cuda")
    p.add_argument("--seed",         type=int,   default=42,
                   help="random seed for reproducibility") # set the initial seed
    p.add_argument("--output_dir",   type=str,
                   default="results")
    p.add_argument("--weight_path",  type=str,
                   default="flow_matching_ct.pt")
    p.add_argument("--cfg_path",     type=str,
                   default="flow_matching_ct.yaml")
    args = p.parse_args()

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device


    angles_deg = torch.linspace(0, 180, args.angles, device=device)
    physics = Tomography(
        img_shape=(1,128,128),
        angles=angles_deg,
        geometry="parallel",
        img_width=128,
        device=device
    )
    ds = CTDataset(tensor_path="/content/drive/MyDrive/CT_dataset_128.pt", measurement_dim=args.angles)
    y_full, x_flat = ds[0]
    x_gt = x_flat.view(1,1,128,128).to(device)

    # Forward + add noise
    y = physics(x_gt)
    if args.noise > 0:
        sigma = torch.mean(torch.abs(y)) * args.noise
        y = y + sigma * torch.randn_like(y)

    x_rec, losses, psnrs = optimise_z(
        physics, 
        y.squeeze(0).squeeze(0).transpose(0,1),
        x_gt,
        args.sampler_steps,
        args.iter_max,
        device,
        args.weight_path,
        args.cfg_path
    )

    mse  = compute_mse(x_rec, x_gt)
    psnr = compute_psnr(x_rec, x_gt)

    tag = f"ang{args.angles}_n{int(args.noise*100)}_seed{args.seed}"
    od  = os.path.join(args.output_dir, tag)
    os.makedirs(od, exist_ok=True)

    plt.imsave(os.path.join(od, "recon.png"),
               x_rec.cpu().numpy().squeeze(), cmap="gray")
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title(tag)
    plt.savefig(os.path.join(od, "loss.png"))
    plt.close()

    # save PSNR curve
    plt.figure(); plt.plot(psnrs)
    plt.xlabel("Iteration"); plt.ylabel("PSNR (dB)"); plt.title(tag + " PSNR")
    plt.savefig(os.path.join(od, "psnr.png")); plt.close()

    summary_csv = os.path.join(args.output_dir, "summary.csv")
    df = pd.DataFrame([{
        "angles": args.angles,
        "noise":  args.noise,
        "mse":    mse,
        "psnr":   psnr
    }])
    df.to_csv(summary_csv, mode="a",
              header=not os.path.exists(summary_csv),
              index=False)

    print(f"[{tag}] done  mse={mse:.4e}  psnr={psnr:.2f}dB")

if __name__ == "__main__":
    main()
