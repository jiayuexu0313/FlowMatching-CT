import torch 
import torch.nn as nn

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 


import time 

import numpy as np 
from tqdm import tqdm 

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

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
torch.manual_seed(1)

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
vf.eval()


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

class WrappedModelReverse(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return -self.model(x, 1 - t)

wrapped_vf = WrappedModel(vf)
wrapped_vf_rev = WrappedModelReverse(vf)

device = "cuda"
dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)

x = dataset[1][0].unsqueeze(0)
x = x.to(device)


### 2. Setup forward operator A 
# simple compressed sensing from 784 -> 256 
# with 1% relative additive Gaussian noise
A = torch.randn(200, 28*28).to(device)/200
#A = torch.eye(28*28).to(device)

y = torch.matmul(x.reshape(x.shape[0], -1), A.T )
y_noise = y + 0.01*torch.mean(torch.abs(y))*torch.randn_like(y).to(device)

x_rec = torch.linalg.solve(A.T @ A, A.T @ y_noise.T)

print("Pseudoinverse: ", x_rec.shape)
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
ax1.set_title("GT")

ax2.imshow(x_rec.reshape(28,28).cpu().numpy(), cmap="gray")
ax2.set_title("Pesudo")

plt.show()

step_size = 0.05

solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class

init_z = torch.randn(1, *[1, 28, 28], device=device) 
z0 = torch.nn.Parameter(torch.clone(init_z))


optimizer = torch.optim.Adam([z0], lr=1e-1)
"""
n_steps = 40
dt = 1. / n_steps

t = torch.tensor(0.0).to(device)

x_sample = init_z.clone().detach()
with torch.no_grad():
	# explicit midpoint rule 
	for _ in range(n_steps):
		model_out_half = wrapped_vf(x_sample, t)
		model_out = wrapped_vf(x_sample + dt/2 *model_out_half, t + dt/2)
		x_sample = x_sample + dt * model_out
		t += dt

t = torch.tensor(0.0).to(device)
xt = x_sample.clone().detach()
with torch.no_grad():
	for _ in range(n_steps):		
		model_out_half = wrapped_vf_rev(xt, t)
		model_out = wrapped_vf_rev(xt + dt/2 *model_out_half, t + dt/2)
		xt = xt + dt * model_out
		t += dt

print("Final t: ", t)
print("Difference of z and x(0): ", torch.mean((xt - init_z)**2))
print("Normalised Difference: ", torch.mean((xt - init_z)**2)/torch.mean(init_z**2))
print("Norm: ", torch.mean(init_z**2))

t = torch.tensor(0.0).to(device)

x_sample_new = xt.clone().detach()
with torch.no_grad():
	# explicit midpoint rule 
	for _ in range(n_steps):
		model_out_half = wrapped_vf(x_sample_new, t)
		model_out = wrapped_vf(x_sample_new + dt/2 *model_out_half, t + dt/2)
		x_sample_new = x_sample_new + dt * model_out
		t += dt

print("Resimulation Errpr: ", torch.mean((x_sample - x_sample_new)**2))

print("x sample: ", x_sample_new.shape)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14,5))

ax1.imshow(init_z[0,0].cpu().numpy())
ax1.set_title("z")

ax2.imshow(x_sample[0,0].cpu().numpy())
ax2.set_title("x(1)")

ax3.imshow(xt[0,0].cpu().numpy())
ax3.set_title("x(0) (reverse from x(1))")

ax4.imshow(x_sample_new[0,0].cpu().numpy())
ax4.set_title("x(1) (resim from x(0))")

plt.show() 
"""
for i in tqdm(range(200)):
	print("\n  --------------- ITERATION: ", i)
	optimizer.zero_grad()
	batch_t = torch.linspace(0., 1., 10).to(device)

	x_sample = solver.sample(time_grid=batch_t, x_init=z0, method='midpoint', step_size=step_size, return_intermediates=False, enable_grad=False)  # sample from the model
	x_sample.requires_grad_(True)
	loss = torch.sum((torch.matmul(x_sample.reshape(x_sample.shape[0], -1), A.T ) - y_noise)**2) + 0.01*torch.mean(z0**2)
	print(loss.item())

	lambda_ = torch.autograd.grad(loss, x_sample, create_graph=True)[0]
	n_steps = 40
	dt = 1. / n_steps
	t = torch.tensor(1.0).to(device)
	xt = x_sample.clone().detach()
	time_start = time.time()
	for _ in range(n_steps):
		# Compute JVP: (\partial f/ \partial x)^T \lambda 
		
		model_out, jvp = torch.autograd.functional.jvp(lambda x: wrapped_vf(x, t), (xt,), (lambda_,))
		xt = xt - dt * model_out
		lambda_ = lambda_ - dt * (-jvp)
		# lamda(t-h) = lambda(t) - h f(lamba(t),t)  # h = 0.01 
		t -= dt

		xt = xt.detach()

	print("Difference of z and x(0): ", torch.mean((xt - z0)**2))
	time_end = time.time()
	print("Time for adjoint pass: ", time_end - time_start,"s")
	z0.grad = lambda_ + 0.005 * z0
	optimizer.step()
	if i % 4 == 0 and i > 0:
		x_sample = torch.clamp(x_sample, 0, 1)

		fig, (ax1, ax2) = plt.subplots(1,2)

		ax1.imshow(x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
		ax2.imshow(x_sample[0,0,:,:].detach().cpu().numpy(), cmap="gray")

		plt.savefig(f"img_{i}.png")
		plt.close()
		#plt.show()


