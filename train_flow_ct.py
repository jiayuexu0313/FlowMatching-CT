import argparse
import ast, yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel
from ct_dataset import CTDataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sampler_steps", type=int, default=5)
    p.add_argument("--subset", type=int, default=0, help="0 = Use all the data; >0 = only use the first N cards")
    p.add_argument("--measurement_dim", type=int, default=200, help="number of CT projection angles")
    p.add_argument("--ct_data_path", type=str, default="CT_dataset_128.pt", help="Downsampled + normalized CT image.pt file (shape [N,128,128])")
    args = p.parse_args()

    device = torch.device(args.device)

    model_cfg = {
        "in_channels": 1,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [8],
        "dropout": 0.0,
        "channel_mult": [1,2,2,4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": True,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
        "max_period": 2.0
    }

    vf = UNetModel(**model_cfg).to(device)
    print("UNet parameter quantity: ", sum(p.numel() for p in vf.parameters()))

    ds = CTDataset(tensor_path=args.ct_data_path, measurement_dim=args.measurement_dim,)
    if args.subset > 0:
        ds.data = ds.data[: args.subset]
        ds.N = len(ds.data)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = Adam(vf.parameters(), lr=1e-4)
    path = AffineProbPath(scheduler=CondOTScheduler())

    print(f"Start training on {device} for {args.epochs} epochs …")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for y, x_flat in tqdm(loader, desc=f"Epoch {epoch}"): # y: [B, measurement_dim, 128] and x_flat: [B,16384] in [0,1]
            x = x_flat.view(-1, 1, 128, 128).to(device)
            x0 = torch.randn_like(x)
            t = torch.rand(x.shape[0], device=device)

            sample = path.sample(t=t, x_0=x0, x_1=x)
            pred = vf(sample.x_t, sample.t)
            loss = ((pred - sample.dx_t) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch} Average loss: {avg_loss:.6f}")

        torch.save(vf.state_dict(), "flow_matching_ct.pt")      
        with open("flow_matching_ct.yaml","w") as f:
            yaml.dump(model_cfg,f)

    print("Finish training. flow_matching_ct.pt, flow_matching_ct.yaml already generated.")

if __name__ == "__main__":
    main()
