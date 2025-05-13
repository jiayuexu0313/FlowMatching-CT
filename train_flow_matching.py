
import numpy as np 
from tqdm import tqdm 

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt 
from torchvision.utils import make_grid

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from models.unet import UNetModel

device = "cuda"

def loss_fn(model, x, marginal_prob_std, marginal_prop_mean, eps=1e-5, T=1.):

    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (T - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
 
    mean = marginal_prop_mean(random_t)
    perturbed_x = mean[:, None, None, None] * x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    
    return loss

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
vf = vf.to(device)



print("Number of Parameters: ", sum([p.numel() for p in vf.parameters()]))



n_epochs = 100
batch_size = 128
lr=1e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())


optimizer = Adam(vf.parameters(), lr=lr)
print("Start Training")
for epoch in range(n_epochs):
    avg_loss = 0.
    num_items = 0
    for x, y in tqdm(data_loader):
        x_1 = x.to(device)    
        x_0 = torch.randn_like(x_1).to(device)
        
        t = torch.rand(x_1.shape[0]).to(device) 
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        loss = torch.pow( vf(path_sample.x_t,path_sample.t) - path_sample.dx_t, 2).mean() 

        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # Print the averaged training loss so far.
    print('Average Loss at epoch {}: {:5f}'.format(epoch, avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(vf.state_dict(), 'flow_matching_mnist.pt')

    with torch.no_grad():
        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                return self.model(x, t)

        wrapped_vf = WrappedModel(vf)

                # step size for ode solver
        step_size = 0.05

        batch_size = 64  # batch size
        times = torch.linspace(0,1,10)  # sample times
        times = times.to(device=device)

        x_init = torch.randn((batch_size, *x_1.shape[1:]), device=device)
        solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
        sol = solver.sample(time_grid=times, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=False)  # sample from the model

        print(sol.shape)
        grid_img = make_grid(sol, nrow=8, normalize=False, padding=2)
        grid_np = grid_img.permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, C]

        plt.figure()
        plt.imshow(grid_np, cmap="gray")
        plt.show()