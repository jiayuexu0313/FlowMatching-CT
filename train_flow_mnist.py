import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

# MNIST 数据及预处理
from torchvision import datasets, transforms

# UNetModel 与 Phase 2 CT 版本一致，但我们要把输入/输出改为 32×32
from models.unet import UNetModel

class MNISTFlowDataset(torch.utils.data.Dataset):
    """
    用于 Unconditional Flow 的 MNIST Dataset。
    只取 MNIST 测试集或训练集，图像通过插值 resize 到 32×32。
    训练时输入 x，也当作 x_1；噪声 x_0 从 N(0,1) 采样。
    """
    def __init__(self, train=True, device="cpu"):
        super().__init__()
        self.device = device
        self.mnist = datasets.MNIST(
            root="./mnist_data",
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),               # float32, [0,1]
                transforms.Normalize((0.5,), (0.5,)) # 归一到 ~[-1,1]
            ])
        )
        self.N = len(self.mnist)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # x: [1,28,28]
        x, _ = self.mnist[idx]
        # 插值到 32×32
        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(32,32),
                                            mode="bilinear", align_corners=False)
        x = x.squeeze(0)  # [1,32,32]
        return x.to(self.device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"],
                   help="设备：'cuda' 或 'cpu'")
    p.add_argument("--epochs", type=int, default=10,
                   help="训练轮数")
    p.add_argument("--batch_size", type=int, default=64,
                   help="批大小")
    p.add_argument("--model_channels", type=int, default=32,
                   help="UNet 初始通道数")
    p.add_argument("--num_res_blocks", type=int, default=2,
                   help="UNet 残差块数")
    p.add_argument("--sampler_steps", type=int, default=5,
                   help="ODE 采样的时间步数")
    args = p.parse_args()

    device = torch.device(args.device)
    # 构建适合 MNIST 32×32 的 UNet Model
    model_cfg = {
        "in_channels": 1,
        "model_channels": args.model_channels,
        "out_channels": 1,
        "num_res_blocks": args.num_res_blocks,
        "attention_resolutions": [],
        "dropout": 0.0,
        "channel_mult": [1,1,1],
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
    print("Parameters:", sum(p.numel() for p in vf.parameters()))

    # 数据集：MNISTTrain
    ds = MNISTFlowDataset(train=True, device=device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = Adam(vf.parameters(), lr=1e-4)
    path = AffineProbPath(scheduler=CondOTScheduler())

    for epoch in range(args.epochs):
        total = 0.0
        for x in tqdm(loader, desc=f"Epoch {epoch}"):
            # x: [B,1,32,32]
            x0 = torch.randn_like(x).to(device)  # 噪声
            t = torch.rand(x.shape[0], device=device)
            sample = path.sample(t=t, x_0=x0, x_1=x)

            pred = vf(sample.x_t, sample.t)
            loss = ((pred - sample.dx_t) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * x.size(0)

        avg = total / len(ds)
        print(f"Epoch {epoch} avg loss: {avg:.6f}")

        # 保存权重 & 采样图
        torch.save(vf.state_dict(), f"flow_mnist_epoch{epoch}.pt")
        with torch.no_grad():
            class Wrapped(ModelWrapper):
                def forward(self, x, t, **kw):
                    return self.model(x, t)
            wrapped = Wrapped(vf)
            solver = ODESolver(velocity_model=wrapped)
            times = torch.linspace(0, 1, args.sampler_steps, device=device)
            x_init = torch.randn((args.batch_size, 1, 32, 32), device=device)
            sol = solver.sample(time_grid=times, x_init=x_init,
                                method="midpoint", step_size=0.05)
            # sol[-1]: [B,1,32,32]，可视化成网格
            grid = make_grid(sol[-1], nrow=8).permute(1, 2, 0).cpu().numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(grid, cmap="gray")
            plt.axis("off")
            plt.savefig(f"sample_mnist_epoch_{epoch}.png")
            plt.close()

if __name__ == "__main__":
    main()
