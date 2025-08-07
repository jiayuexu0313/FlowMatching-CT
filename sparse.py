import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # faster backend for remote
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from deepinv.physics import Tomography
from ct_dataset import CTDataset
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel

NOISE = 0.05
STEPS = 6
DEVICE_DEF = "cuda"

def compute_mse(x_rec, x_gt):
    return torch.mean((x_rec - x_gt)**2).item()

def compute_psnr(x_rec, x_gt):
    mse = compute_mse(x_rec, x_gt)
    return 20 * np.log10(1.0 / np.sqrt(mse))

def build_physics(device, num_angles):
    angles = torch.linspace(0, 180, num_angles, device=device, dtype=torch.float32)
    return Tomography(
        img_shape=(1,128,128),
        angles=angles,
        geometry="parallel",
        img_width=128,
        device=device)

def load_model(device, cfg_path, weight_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    vf = UNetModel(**cfg).to(device).eval()
    vf.load_state_dict(torch.load(weight_path, map_location=device))
    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw): return vf(x, t)
    return Wrapped(vf)

def get_data(device, num_angles):
    physics = build_physics(device, num_angles)
    ds = CTDataset(tensor_path="CT_dataset_128.pt", measurement_dim=num_angles)
    y_full, x_flat = ds[0]
    x_gt = x_flat.view(1,1,128,128).to(device)

    y = physics(x_gt)
    sigma = torch.mean(torch.abs(y)) * NOISE
    y = y + sigma * torch.randn_like(y)
    y = y.squeeze(0).squeeze(0).transpose(0,1)
    return physics, y, x_gt

def optimise_z_unrolled(physics, y, x_gt, wrapped, iter_max, device, z):
    solver = ODESolver(velocity_model=wrapped)
    opt  = torch.optim.Adam([z], lr=1e-1)
    losses, psnrs = [], []
    for i in range(iter_max):
        opt.zero_grad()
        t_grid = torch.linspace(0,1,STEPS,device=device)
        x_hat  = solver.sample(
            time_grid=t_grid,
            x_init=z,
            method="midpoint",
            step_size=1/STEPS,
            enable_grad=True)
        ypred = physics(x_hat).squeeze(0).squeeze(0).transpose(0,1)
        loss  = ((ypred - y)**2).sum() + 0.01*(z**2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            rec_temp = x_hat.clamp(0,1).detach()
            psnrs.append(compute_psnr(rec_temp, x_gt))
        if (i+1) % 200 == 0:
            print(f"[Unrolled] iter {i+1}/{iter_max}: loss={losses[-1]:.3e}, PSNR={psnrs[-1]:.2f}")
    return x_hat.clamp(0,1).detach(), losses, psnrs

def optimise_z_dflow(physics, y, x_gt, wrapped, iter_max, device, z):
    solver = ODESolver(velocity_model=wrapped)
    opt = torch.optim.LBFGS(
        [z],
        lr=1.0,
        max_iter=1,
        history_size=100,
        line_search_fn='strong_wolfe')
    t_grid = torch.linspace(0,1,STEPS,device=device)
    losses, psnrs = [], []
    for i in range(iter_max):
        def closure():
            opt.zero_grad()
            x_hat = solver.sample(
                time_grid=t_grid,
                x_init=z,
                method="midpoint",
                step_size=1/STEPS,
                enable_grad=True)
            ypred = physics(x_hat).squeeze(0).squeeze(0).transpose(0,1)
            loss  = ((ypred - y)**2).sum() + 0.01*(z**2).mean()
            loss.backward()
            return loss
        loss = opt.step(closure)
        with torch.no_grad():
            x_hat = solver.sample(
                time_grid=t_grid,
                x_init=z,
                method="midpoint",
                step_size=1/STEPS,
                enable_grad=False).clamp(0,1)
            losses.append(loss.item())
            psnrs.append(compute_psnr(x_hat, x_gt))
        if (i+1) % 200 == 0:
            print(f"[D-Flow] iter {i+1}/{iter_max}: loss={losses[-1]:.3e}, PSNR={psnrs[-1]:.2f}")
    with torch.no_grad():
        x_hat = solver.sample(
            time_grid=t_grid,
            x_init=z,
            method="midpoint",
            step_size=1/STEPS,
            enable_grad=False).clamp(0,1)
    return x_hat.detach(), losses, psnrs

def optimise_z_adjoint(physics, y, x_gt, wrapped, iter_max, device, z):
    solver = ODESolver(velocity_model=wrapped)
    opt    = torch.optim.Adam([z], lr=1e-1)
    losses, psnrs = [], []
    for i in range(iter_max):
        opt.zero_grad()
        t_grid = torch.linspace(0,1,STEPS,device=device)
        x_hat  = solver.sample(
            time_grid=t_grid,
            x_init=z,
            method="midpoint",
            step_size=1/STEPS,
            enable_grad=False)
        x_hat.requires_grad_(True)
        ypred = physics(x_hat).squeeze(0).squeeze(0).transpose(0,1)
        loss  = ((ypred - y)**2).sum() + 0.01*(z**2).mean()
        lam   = torch.autograd.grad(loss, x_hat, create_graph=True)[0]
        n_steps, dt = 20, 1/20
        t_val = torch.tensor(1., device=device)
        xt    = x_hat.clone().detach()
        for _ in range(n_steps):
            model_out, jvp = torch.autograd.functional.jvp(lambda v: wrapped(v, t_val), (xt,), (lam,))
            xt  = xt  - dt*model_out
            lam = lam - dt*(-jvp)
            t_val -= dt
            xt = xt.detach()
        z.grad = lam + 0.01*z
        opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            rec_temp = x_hat.clamp(0,1).detach()
            psnrs.append(compute_psnr(rec_temp, x_gt))
        if (i+1) % 200 == 0:
            print(f"[Adjoint] iter {i+1}/{iter_max}: loss={losses[-1]:.3e}, PSNR={psnrs[-1]:.2f}")
    return x_hat.clamp(0,1).detach(), losses, psnrs

def run_once(seed, method, init, cfg_path, weight_path, iter_max, device, output_dir, num_angles):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device=="cuda": torch.cuda.manual_seed_all(seed)

    physics, y, x_gt = get_data(device, num_angles)

    detector_resolution = np.sqrt(128**2 + 128**2)
    undersampling = (num_angles * detector_resolution) / (128*128)
    print(f"[seed={seed}] num_angles={num_angles}, undersampling={undersampling:.3f}")

    wrapped = load_model(device, cfg_path, weight_path)

    # FBP init
    fbp = physics.A_dagger(y.transpose(0,1).unsqueeze(0).unsqueeze(0))
    fbp_np = fbp.clamp(0,1).cpu().numpy().squeeze()

    # z0 init
    if init=="dflow":
        t_rev = torch.linspace(1,0,STEPS,device=device)
        z0    = ODESolver(velocity_model=wrapped).sample(
            time_grid=t_rev,
            x_init=fbp,
            method="midpoint",
            step_size=1/STEPS,
            enable_grad=False).detach().requires_grad_(True)
    else:
        z0 = torch.randn_like(x_gt, device=device, requires_grad=True)

    # method：
    if method=="unrolled":
        if device=="cuda":
            torch.cuda.reset_peak_memory_stats()
        rec, losses, psnrs = optimise_z_unrolled(physics, y, x_gt, wrapped, iter_max, device, z0)
        if device=="cuda":
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[seed={seed}] Unrolled Peak GPU memory: {peak:.2f} GB")

    elif method=="dflow":
        if device=="cuda":
           torch.cuda.reset_peak_memory_stats()
        rec, losses, psnrs = optimise_z_dflow(physics, y, x_gt, wrapped, iter_max, device, z0)
        if device=="cuda":
           peak = torch.cuda.max_memory_allocated()/(1024**3)
           print(f"[seed={seed}] D-Flow Peak GPU memory: {peak:.2f} GB")

    else:  # adjoint
        if device=="cuda":
            torch.cuda.reset_peak_memory_stats()
        rec, losses, psnrs = optimise_z_adjoint(physics, y, x_gt, wrapped, iter_max, device, z0)
        if device=="cuda":
            peak = torch.cuda.max_memory_allocated()/(1024**3)
            print(f"[seed={seed}] Adjoint Peak GPU memory: {peak:.2f} GB")

    od_seed = os.path.join(output_dir,f"{method}_{init}_num{num_angles}",f"seed{seed}")
    os.makedirs(od_seed, exist_ok=True)
    plt.figure(); plt.plot(losses); plt.xlabel('Iteration'); plt.ylabel('Loss')
    plt.title(f'{method}_{init} loss'); plt.savefig(os.path.join(od_seed,'loss.png')); plt.close()
    plt.figure(); plt.plot(psnrs); plt.xlabel('Iteration'); plt.ylabel('PSNR (dB)')
    plt.title(f'{method}_{init} PSNR'); plt.savefig(os.path.join(od_seed,'psnr.png')); plt.close()

    gt_np  = x_gt.cpu().numpy().squeeze()
    rec_np = rec.cpu().numpy().squeeze()
    out    = np.concatenate([gt_np, fbp_np, rec_np], axis=1)
    od_seed = os.path.join(output_dir,f"{method}_{init}_num{num_angles}",f"seed{seed}")
    os.makedirs(od_seed, exist_ok=True)
    plt.imsave(os.path.join(od_seed, "images.png"), out, cmap="gray")

    pval = compute_psnr(rec, x_gt)
    sval = ssim(gt_np, rec_np, data_range=1.0)
    return pval, sval, peak

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["unrolled","dflow","adjoint"], required=True)
    p.add_argument("--init", choices=["random","dflow"], default="random")
    p.add_argument("--seeds", type=str, required=True)
    p.add_argument("--iter_max", type=int, default=200)
    p.add_argument("--cfg_path", type=str, default="flow_matching_ct.yaml")
    p.add_argument("--weight_path",type=str, default="flow_matching_ct.pt")
    p.add_argument("--device", choices=["cpu","cuda"], default=DEVICE_DEF)
    p.add_argument("--num_angles", type=int, default=60, help="number of sparse angles, such as 90, 60, 30")
    p.add_argument("--output_dir", type=str, default="results", help="output path")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    psnrs = []
    ssims = []
    mems = []

    for sd in seeds:
        pval, sval, peak = run_once(
            seed=sd,
            method=args.method,
            init=args.init,
            cfg_path=args.cfg_path,
            weight_path=args.weight_path,
            iter_max=args.iter_max,
            device=args.device,
            output_dir=args.output_dir,
            num_angles=args.num_angles)
        psnrs.append(pval); ssims.append(sval)
        mems.append(peak or float('nan'))
        print(f"[seed={sd}] PSNR={pval:.2f}dB  SSIM={sval:.4f}")

    psnrs = np.array(psnrs); ssims = np.array(ssims)
    print(f"=== {args.method} + {args.init} ===")
    print(f"PSNR mean={psnrs.mean():.2f}  std={psnrs.std():.2f}")
    print(f"SSIM mean={ssims.mean():.4f}  std={ssims.std():.4f}")

    df = pd.DataFrame({
        "seed": seeds,
        "psnr": psnrs,
        "ssim": ssims,
        "memory_gb": mems})
    summary_path = os.path.join(args.output_dir, f"summary_{args.method}_{args.init}.csv")
    df.to_csv(summary_path, index=False)
    print("Summary saved to", summary_path)

if __name__=="__main__":
    main()
