# import argparse
# import torch
# import numpy as np
# from tqdm import tqdm
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid

# # flow_matching
# from flow_matching.path.scheduler import CondOTScheduler
# from flow_matching.path import AffineProbPath
# from flow_matching.solver import ODESolver
# from flow_matching.utils import ModelWrapper

# from models.unet import UNetModel
# from ct_dataset import CTDataset

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
#     p.add_argument("--epochs", type=int, default=30)
#     p.add_argument("--batch_size", type=int, default=4)
#     p.add_argument("--model_channels", type=int, default=32)
#     p.add_argument("--num_res_blocks", type=int, default=2)
#     p.add_argument("--sampler_steps", type=int, default=5)
#     p.add_argument("--subset", type=int, default=0, 
#                    help="0=use all, >0 use first N samples")
#     args = p.parse_args()

#     device = torch.device(args.device)
#     model_cfg = {
#         "in_channels": 1,
#         "model_channels": args.model_channels,
#         "out_channels": 1,
#         "num_res_blocks": args.num_res_blocks,
#         "attention_resolutions": [],
#         "dropout": 0.0,
#         "channel_mult": [1, 1, 1],
#         "conv_resample": False,
#         "dims": 2,
#         "num_classes": None,
#         "use_checkpoint": True,
#         "num_heads": 1,
#         "num_head_channels": -1,
#         "num_heads_upsample": -1,
#         "use_scale_shift_norm": True,
#         "resblock_updown": False,
#         "use_new_attention_order": True,
#         "with_fourier_features": False,
#         "max_period": 2.0
#     }

#     vf = UNetModel(**model_cfg).to(device)
#     print("Parameters:", sum(p.numel() for p in vf.parameters()))

#     # CT data
#     ds = CTDataset("/content/drive/MyDrive/CT_dataset_128.pt", measurement_dim=200)
#     if args.subset > 0:
#         ds.data = ds.data[: args.subset]
#         ds.N = len(ds.data)
#     loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

#     optimizer = Adam(vf.parameters(), lr=1e-4)
#     path = AffineProbPath(scheduler=CondOTScheduler())

#     print(f"Start training on {device} for {args.epochs} epochs …")
#     for epoch in range(args.epochs):
#         total = 0.0
#         for y, x_flat in tqdm(loader, desc=f"Epoch {epoch}"):
#             x = x_flat.view(-1,1,128,128).to(device)
#             x0 = torch.randn_like(x)
#             t = torch.rand(x.shape[0], device=device)
#             sample = path.sample(t=t, x_0=x0, x_1=x)
#             pred = vf(sample.x_t, sample.t)
#             loss = ((pred - sample.dx_t)**2).mean()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total += loss.item() * x.size(0)

#         avg = total / len(ds)
#         print(f"Epoch {epoch} avg loss: {avg:.6f}")
#         torch.save(vf.state_dict(), "flow_matching_ct.pt")

#         # Median sampling is visualized once per epoch
#         with torch.no_grad():
#             class Wrapped(ModelWrapper):
#                 def forward(self, x, t, **kw): return self.model(x, t)
#             wrapped = Wrapped(vf)
#             solver = ODESolver(velocity_model=wrapped)
#             times = torch.linspace(0,1,args.sampler_steps,device=device)
#             x_init = torch.randn((args.batch_size,1,128,128), device=device)
#             sol = solver.sample(time_grid=times, x_init=x_init,
#                                 method="midpoint", step_size=0.05)
#             grid = make_grid(sol, nrow=args.batch_size).permute(1,2,0).cpu().numpy()
#             plt.figure(figsize=(4,4))
#             #grid_np = np.clip(grid_np, 0, 1)
#             plt.imshow(grid, cmap="gray")
#             plt.axis("off")
#             plt.show()

# if __name__=="__main__":
#     main()



# modify & add log loss:
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
    p.add_argument("--model_channels", type=int, default=32)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--sampler_steps", type=int, default=5)
    p.add_argument("--subset", type=int, default=0,
                   help="0 = Use all the data; >0 = only use the first N cards")
    p.add_argument("--measurement_dim", type=int, default=200,
                   help="number of CT projection angles")
    #Add a new line channel_mult：
    p.add_argument("--channel_mult", type=str, default="[1,1,1]",
                   help="list of multipliers, e.g. \"[1,2,4]\"")
    p.add_argument("--ct_data_path", type=str,
                   default="/content/drive/MyDrive/CT_dataset_128.pt",
                   help="Downsampled + normalized CT image.pt file (shape [N,128,128])")
    args = p.parse_args()

    device = torch.device(args.device)

    model_cfg = {
        "in_channels": 1,
        "model_channels": args.model_channels,
        "out_channels": 1,
        "num_res_blocks": args.num_res_blocks,
        "attention_resolutions": [],
        "dropout": 0.0,
        # "channel_mult": [1, 1, 1], change to [1,2,2,4]
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
    # Add a new line: Read channel_mult
    model_cfg["channel_mult"] = ast.literal_eval(args.channel_mult)
    vf = UNetModel(**model_cfg).to(device)
    print("UNet parameter quantity: ", sum(p.numel() for p in vf.parameters()))

    ds = CTDataset(
        tensor_path=args.ct_data_path,
        measurement_dim=args.measurement_dim,
    )
    if args.subset > 0:
        ds.data = ds.data[: args.subset]
        ds.N = len(ds.data)

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=2)

    optimizer = Adam(vf.parameters(), lr=1e-4)
    path = AffineProbPath(scheduler=CondOTScheduler())

    # Write the average loss of each epoch
    loss_log = open("train_loss.txt", "w")
    loss_log.write("epoch,avg_loss\n")

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
        loss_log.write(f"{epoch},{avg_loss:.6f}\n")
        loss_log.flush()

        torch.save(vf.state_dict(), "flow_matching_ct.pt")
        # Add a new line: save model_cfg        
        with open("flow_matching_ct.yaml","w") as f:
            yaml.dump(model_cfg,f)

        # do a unconditional sampling and save the image sample_epoch_{epoch}.png
        with torch.no_grad():
            class Wrapped(ModelWrapper):
                def forward(self, x, t, **kw):
                    return self.model(x, t)
            wrapped = Wrapped(vf)
            solver = ODESolver(velocity_model=wrapped)

            times = torch.linspace(0, 1, args.sampler_steps, device=device)
            x_init = torch.randn((args.batch_size, 1, 128, 128), device=device)
            sol = solver.sample(time_grid=times,
                                x_init=x_init,
                                method="midpoint",
                                step_size=0.05)
            sample_imgs = sol[-1].clamp(0, 1)

            grid = make_grid(sample_imgs, nrow=args.batch_size).permute(1, 2, 0).cpu().numpy()
            # plt.figure(figsize=(4, 4))
            # plt.imshow(grid, cmap="gray")
            # plt.axis("off")
            # plt.savefig(f"sample_epoch_{epoch}.png", bbox_inches="tight", pad_inches=0)
            # plt.close()

            # Add a new part:
            plt.imsave(f"sample_epoch/sample_epoch_{epoch}.png", grid, cmap="gray")
            # Nearest neighbor visualization (Take the first one in the batch)
            sample0 = sample_imgs[0].view(-1).cpu()
            data_flat = ds.data.view(ds.N,-1)
            dists = torch.mean((data_flat - sample0)**2, dim=1)
            top5 = torch.topk(dists, k=5, largest=False).indices.tolist()
            neigh = ds.data[top5]  # [5,128,128]
            kn_grid = make_grid(neigh.unsqueeze(1), nrow=5)\
                        .permute(1,2,0).cpu().numpy()
            plt.imsave(f"neighbors_eopch/neighbors_epoch_{epoch}.png", kn_grid, cmap="gray")

    loss_log.close()
    print("Finish training. train_loss.txt, flow_matching_ct.pt, flow_matching_ct.yaml, sample_epoch_{epoch}.png and neighbors_epoch_{epoch}.png already generated.")

if __name__ == "__main__":
    main()
