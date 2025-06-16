import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from deepinv.physics import Tomography
from ct_dataset import CTDataset

def compute_mse(x_rec: torch.Tensor, x_gt: torch.Tensor) -> float:
    return torch.mean((x_rec - x_gt)**2).item()

def compute_psnr(x_rec: torch.Tensor, x_gt: torch.Tensor) -> float:
    mse = compute_mse(x_rec, x_gt)
    return 20 * np.log10(1.0 / np.sqrt(mse))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--angles",     type=int,   required=True)
    p.add_argument("--noise",      type=float, default=0.0)
    p.add_argument("--device",     choices=["cpu","cuda"], default="cuda")
    p.add_argument("--output_dir", type=str,   default="results_fbp")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    ds = CTDataset(tensor_path="/content/drive/MyDrive/CT_dataset_128.pt", measurement_dim=args.angles)
    y_full, x_flat = ds[0]
    x_gt = x_flat.view(1,1,128,128).to(device)

    angles_deg = torch.linspace(0, 180, args.angles, device=device)
    physics = Tomography(
        angles=angles_deg,
        img_shape=(1, 128, 128),
        circle=False,
        img_width=128,
        device=device
    )
    y = physics.A(x_gt)
    if args.noise > 0:
        sigma = torch.mean(torch.abs(y)) * args.noise
        y = y + sigma * torch.randn_like(y)

    x_rec = physics.A_dagger(y).clamp(0,1).detach()

    mse  = compute_mse(x_rec, x_gt)
    psnr = compute_psnr(x_rec, x_gt)

    tag = f"ang{args.angles}_n{int(args.noise*100)}"
    od  = os.path.join(args.output_dir, tag)
    os.makedirs(od, exist_ok=True)

    plt.imsave(os.path.join(od, "recon_fbp.png"), x_rec.cpu().numpy().squeeze(), cmap="gray")

    summary_csv = os.path.join(args.output_dir, "summary_fbp.csv")
    df = pd.DataFrame([{
        "angles": args.angles,
        "noise":  args.noise,
        "mse":    mse,
        "psnr":   psnr
    }])
    df.to_csv(summary_csv, mode="a", header=not os.path.exists(summary_csv), index=False)

    print(f"[FBP] angles={args.angles} noise={args.noise}  mse={mse:.4e}  psnr={psnr:.2f}dB")

if __name__ == "__main__":
    main()
