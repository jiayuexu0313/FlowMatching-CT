import os
import argparse
import yaml
import math
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver
from models.unet import UNetModel

def main():
    parser = argparse.ArgumentParser(description="Visualize flow-based sampling process")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to the flow-matching model YAML config file")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to the trained model weights (.pt)")
    parser.add_argument("--sampler_steps", type=int, default=50, help="Number of ODE sampling steps (time grid length)")
    parser.add_argument("--img_size", type=int, default=128, help="Height and width of the square images")
    parser.add_argument("--output_dir", type=str, default="visualization", help="Directory to save intermediate images and grid")
    parser.add_argument( "--device", type=str, choices=["cpu","cuda"], default="cuda", help="Compute device")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    vf = UNetModel(**model_cfg).to(device).eval()
    vf.load_state_dict(torch.load(args.weight_path, map_location=device))

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return vf(x, t)
    wrapped = Wrapped(vf)
    solver = ODESolver(velocity_model=wrapped)

    x_init = torch.randn((1, model_cfg.get('in_channels', 1), args.img_size, args.img_size), device=device)

    # Create time grid from noise (t=0) -> data (t=1)
    time_grid = torch.linspace(0, 1, args.sampler_steps, device=device)

    # Sample with intermediates
    with torch.no_grad():
        traj = solver.sample(
            x_init=x_init,
            time_grid=time_grid,
            method="midpoint",
            step_size=1.0/args.sampler_steps,
            return_intermediates=True)  # shape: [T, B, C, H, W]
    traj = traj.clamp(0, 1).cpu()

    # Save each intermediate frame
    for i in range(traj.shape[0]):
        img = traj[i, 0, 0]  # first batch, first channel
        out_path = os.path.join(args.output_dir, f"step_{i:03d}.png")
        plt.imsave(out_path, img.numpy(), cmap="gray")

    # Save full trajectory grid
    n_steps = traj.shape[0]
    n_col = int(math.ceil(math.sqrt(n_steps)))
    grid = make_grid(traj[:, 0], nrow=n_col).permute(1, 2, 0)
    grid_path = os.path.join(args.output_dir, "sampling_grid.png")
    plt.imsave(grid_path, grid.numpy())

    print(f"Saved {n_steps} intermediate frames to '{args.output_dir}' and grid to '{grid_path}'.")

if __name__ == "__main__":
    main()