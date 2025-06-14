# import argparse, os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import numpy as np
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from torch.optim import Adam

# from flow_matching.solver import ODESolver
# from flow_matching.utils import ModelWrapper
# from models.unet import UNetModel
# from ct_dataset import CTDataset

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
#     p.add_argument("--sampler_steps", type=int, default=5)
#     p.add_argument("--iter_max",    type=int, default=200)
#     p.add_argument("--model_channels", type=int, default=32)
#     p.add_argument("--num_res_blocks", type=int, default=2)
#     args = p.parse_args()

#     device = torch.device(args.device)
#     model_cfg = {
#         "in_channels": 1,
#         "model_channels": args.model_channels,
#         "out_channels": 1,
#         "num_res_blocks": args.num_res_blocks,
#         "attention_resolutions": [],
#         "dropout": 0.0,
#         "channel_mult": [1,1,1],
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

#     # 1. load model
#     vf = UNetModel(**model_cfg)
#     vf.load_state_dict(torch.load("flow_matching_ct.pt", map_location=device))
#     vf = vf.to(device)

#     class Wrapped(ModelWrapper):
#         def forward(self, x, t, **kw): return self.model(x, t)
#     wrapped = Wrapped(vf)
#     solver = ODESolver(velocity_model=wrapped)

#     # 2. prepare data + measurement
#     ds = CTDataset("/content/drive/MyDrive/CT_dataset_128.pt", measurement_dim=200)
#     x, y = ds[0][1].view(1,1,128,128), ds[0][0].view(1,-1)
#     x, y = x.to(device), y.to(device)
#     A = ds.A.to(device)

#     # 3. latent z optimization
#     z = torch.randn_like(x, requires_grad=True, device=device)
#     opt = Adam([z], lr=1e-1)

#     losses = []
#     for i in tqdm(range(args.iter_max), desc="Reconstruction"):
#         opt.zero_grad()
#         t_grid = torch.linspace(0,1,args.sampler_steps,device=device)
#         x_hat = solver.sample(
#             time_grid=t_grid,
#             x_init=z,
#             method="midpoint",
#             step_size=0.05,
#             enable_grad=True
#         )
#         loss = ((x_hat.view(-1,128*128) @ A.t() - y)**2).sum() + 0.01*(z**2).mean()
#         losses.append(loss.item())
#         loss.backward()
#         opt.step()
#         torch.cuda.empty_cache()

#         if i % 4 == 0 and i > 0:
#             gt_np = x.detach().cpu().squeeze().numpy()
#             xh_np = x_hat.clamp(0,1).detach().cpu().squeeze().numpy()
#             gt_np = np.clip(gt_np, 0, 1)
#             out = np.concatenate([gt_np, xh_np], axis=1)
#             plt.imsave(f"img_mid_{i}.png", out, cmap="gray")
    
#     # 4. plot & save loss curve
#     plt.figure()
#     plt.plot(losses, label="midpoint loss")
#     plt.xlabel("Iteration"); plt.ylabel("Loss")
#     plt.title("Midpoint Latent Optimization Loss")
#     plt.legend()
#     plt.savefig("loss_mid.png")
#     plt.close()

# if __name__=="__main__":
#     main()





# Radon:

import argparse, os
import ast
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel

from ct_dataset import CTDataset
from deepinv.physics import Tomography
#from skimage.transform import iradon

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"],
                   help="device:'cuda' or 'cpu'")
    p.add_argument("--sampler_steps", type=int, default=5,
                   help=" number of time steps used for ODE sampling")
    p.add_argument("--iter_max", type=int, default=200,
                   help="number of iterations for latent variable optimization")
    p.add_argument("--model_channels", type=int, default=32,
                   help=" initial number of channels in the UNet model")
    p.add_argument("--num_res_blocks", type=int, default=2,
                   help="number of Residual blocks in UNet")
    p.add_argument("--measurement_dim", type=int, default=200,
                   help="number of projection angles of a finite Angle CT")
    # Add a new line channel_mult:
    p.add_argument("--channel_mult", type=str, default="[1,1,1]",
                   help="U-Net list of multipliers, e.g. \"[1,2,4]\"")
    p.add_argument("--ct_data_path", type=str,
                   default="/content/drive/MyDrive/CT_dataset_128.pt",
                   help="CT image tension.pt file path (shape=[N,128,128])")
    args = p.parse_args()

    device = torch.device(args.device)

    # model_cfg = {
    #     "in_channels": 1,
    #     "model_channels": args.model_channels,
    #     "out_channels": 1,
    #     "num_res_blocks": args.num_res_blocks,
    #     "attention_resolutions": [],
    #     "dropout": 0.0,
    #     # "channel_mult": [1, 1, 1], change to [1,2,4]
    #     "conv_resample": False,
    #     "dims": 2,
    #     "num_classes": None,
    #     "use_checkpoint": True,
    #     "num_heads": 1,
    #     "num_head_channels": -1,
    #     "num_heads_upsample": -1,
    #     "use_scale_shift_norm": True,
    #     "resblock_updown": False,
    #     "use_new_attention_order": True,
    #     "with_fourier_features": False,
    #     "max_period": 2.0
    # }
    # # Add a new line: Read channel_mult
    # model_cfg["channel_mult"] = ast.literal_eval(args.channel_mult)


    #use new pt：
    model_cfg = {
    "in_channels": 1,
    "model_channels": 32,
    "out_channels": 1,
    "num_res_blocks": 2,
    "attention_resolutions": [8],
    "dropout": 0.0,
    "channel_mult": [1, 2, 2, 4],
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

    vf = UNetModel(**model_cfg)
    vf.load_state_dict(torch.load("flow_matching_ct.pt", map_location=device))
    vf = vf.to(device).eval()

    class Wrapped(ModelWrapper):
        def forward(self, x, t, **kw):
            return self.model(x, t)

    wrapped = Wrapped(vf)
    solver = ODESolver(velocity_model=wrapped)

    ds = CTDataset(
        tensor_path=args.ct_data_path,
        measurement_dim=args.measurement_dim
    )

    y_cpu, x_flat = ds[0]
    # Construct the Radon operator on the GPU
    angles_gpu = torch.linspace(0, np.pi, args.measurement_dim,
                                device=device, dtype=torch.float32)
    physics_gpu = Tomography(
        img_shape=(1, 128, 128),
        angles=angles_gpu,
        geometry="parallel",
        img_width=128,
        device=device
    )

    x = x_flat.view(1, 1, 128, 128).to(device)  
    # using GPU physics to calculate sinogram y; shape [1, num_det, measurement_dim]
    y_full = physics_gpu(x)
    y = y_full.squeeze(0).squeeze(0).transpose(0, 1).contiguous()

    #  FBP baseline
    angles_fbp = torch.linspace(0, 180, 60, device=device, dtype=torch.float32)
    physics_fbp = Tomography(
        angles=angles_fbp,
        img_shape=(1, 128, 128),
        circle=False,
        img_width=128,
        device=device
    )
    y_fbp = physics_fbp.A(x)  
    print("FBP sinogram shape:", y_fbp.shape)
    x_fbp = physics_fbp.A_dagger(y_fbp)  
    
    # can change: add 5% relative noise
    # mean_abs = torch.mean(torch.abs(y_fbp))
    # y_noise = y_fbp + 0.05 * mean_abs * torch.randn_like(y_fbp)
    # x_fbp = physics_fbp.A_dagger(y_noise)
    fbp_np = x_fbp.clamp(0,1).detach().cpu().squeeze().numpy()
    plt.imsave("fbp_mid.png", fbp_np, cmap="gray")


    z = torch.randn_like(x, requires_grad=True, device=device)
    opt = torch.optim.Adam([z], lr=1e-1)

    losses = []
    for i in tqdm(range(args.iter_max), desc="Midpoint Latent Opt"):
        opt.zero_grad()

        # ODESolver (Midpoint) samples from z to obtain x_hat
        t_grid = torch.linspace(0, 1, args.sampler_steps, device=device)
        x_hat = solver.sample(
            time_grid=t_grid,
            x_init=z,
            method="midpoint",
            step_size=0.05,
            enable_grad=True #modify from false to true
        )
        x_hat.requires_grad_(True)
        # can be modified to prevent memory explosion:
        # with torch.no_grad():
        #     x_hat = solver.sample(
        #         time_grid=t_grid,
        #         x_init=z,
        #         method="midpoint",
        #         step_size=0.05,
        #         enable_grad=True
        #     )
        # x_hat = x_hat.clone().detach().requires_grad_(True)

        # calaulate sinogram y_pred
        y_pred_full = physics_gpu(x_hat)  # shape [1, num_det, measurement_dim]
        y_pred = y_pred_full.squeeze(0).squeeze(0).transpose(0, 1).contiguous()  # [measurement_dim, num_det]

        data_fidelity = ((y_pred - y) ** 2).sum()
        reg = 0.01 * (z ** 2).mean()
        loss = data_fidelity + reg
        losses.append(loss.item())

        loss.backward()
        opt.step()
        torch.cuda.empty_cache()
        # can be modified to prevent memory explosion:
        # with torch.no_grad():
        #     torch.cuda.empty_cache()

        # Save the reconstructed comparison map every 10 steps
        if i % 10 == 0 and i > 0:
            gt_np = x.detach().cpu().squeeze().numpy()
            xh_np = x_hat.clamp(0, 1).detach().cpu().squeeze().numpy()
            out = np.concatenate([gt_np, xh_np], axis=1)
            plt.imsave(f"mid_images/img_mid_{i}.png", out, cmap="gray")

    plt.figure()
    plt.plot(losses, label="midpoint loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Midpoint Latent Optimization Loss")
    plt.legend()
    plt.savefig("mid_images/loss_mid.png")
    plt.close()

if __name__ == "__main__":
    main()