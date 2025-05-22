import torch 
import torch.nn as nn

# from torchvision.datasets import MNIST
# import torchvision.transforms as transforms
# CT_baseline replace with:
from ct_dataset import CTDataset

import matplotlib.pyplot as plt 


from tqdm import tqdm 

import numpy as np 
from tqdm import tqdm 

from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from models.unet import UNetModel

"""
In this script the goal is to find the latent vector z which corresponds to an 
image x, by minimizing 
	min_z || ODESolve(z, 1 -> 0) - x || 

"""
torch.manual_seed(0)

device = "cuda"


model_cfg = {"in_channels": 1,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.0,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
        "max_period": 2.0}


vf = UNetModel(**model_cfg)
vf.load_state_dict(torch.load("flow_matching_mnist.pt"))
vf = vf.to(device)


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

wrapped_vf = WrappedModel(vf)

device = "cuda"

# dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
# x = dataset[1][0].unsqueeze(0)
# x = x.to(device)
# CT_baseline replace with:
dataset = CTDataset('/content/drive/MyDrive/CT_dataset_128.pt')
x = dataset[0][1].reshape(1, 1, 128, 128).to(device)
y_noise = dataset[0][0].unsqueeze(0).to(device)
A = dataset.A.to(device)



### 2. Setup forward operator A 
# simple compressed sensing from 784 -> 256 
# with 1% relative additive Gaussian noise
A = torch.randn(256, 28*28).to(device)/256
#A = torch.eye(28*28).to(device)

y = torch.matmul(x.reshape(x.shape[0], -1), A.T )
y_noise = y + 0.01*torch.mean(torch.abs(y))*torch.randn_like(y).to(device)


step_size = 0.05

solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class

# init_z = torch.randn(1, *[1, 28, 28], device=device) 
# CT_baseline replace with:
init_z = torch.randn(1, 1, 128, 128, device=device)

z0 = torch.nn.Parameter(torch.clone(init_z))


optimizer = torch.optim.Adam([z0], lr=1e-1)

for i in tqdm(range(200)):
	print("\n  --------------- ITERATION: ", i)
	optimizer.zero_grad()
	batch_t = torch.linspace(0., 1., 10).to(device)

	x_sample = solver.sample(time_grid=batch_t, x_init=z0, method='midpoint', step_size=step_size, return_intermediates=False, enable_grad=True)  # sample from the model
    
	print(x_sample.shape)
	
	loss = torch.sum((torch.matmul(x_sample.reshape(x_sample.shape[0], -1), A.T ) - y_noise)**2) + 0.01*torch.mean(z0**2)
	print(loss.item())

	loss.backward()
	optimizer.step()

	if i % 4 == 0 and i > 0:
		x_sample = torch.clamp(x_sample, 0, 1)
		print(x_sample.shape)

		fig, (ax1, ax2) = plt.subplots(1,2)

		ax1.imshow(x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
		ax2.imshow(x_sample[0,0,:,:].detach().cpu().numpy(), cmap="gray")

		plt.savefig(f"img_{i}.png")
		plt.close()
		#plt.show()


