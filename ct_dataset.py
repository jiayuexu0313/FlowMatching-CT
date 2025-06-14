# import torch
# from torch.utils.data import Dataset

# class CTDataset(Dataset):
#     def __init__(self, tensor_path='/content/drive/MyDrive/CT_dataset_128.pt', measurement_dim=200, seed=42):
#         self.data = torch.load(tensor_path)  # [N, 128, 128]
#         self.N = self.data.shape[0]
#         self.input_dim = 128 * 128
#         self.measurement_dim = measurement_dim

#         torch.manual_seed(seed)
#         self.A = torch.randn(self.measurement_dim, self.input_dim)

#     def __len__(self):
#         return self.N

#     def __getitem__(self, idx):
#         x = self.data[idx].view(-1)  # 展平为 [16384]
#         y = torch.matmul(self.A, x)  # A·x
#         return y, x




# Radon:
import torch
from torch.utils.data import Dataset
import numpy as np

import deepinv as dinv
from deepinv.physics import Tomography

class CTDataset(Dataset):
    """
    CPU version:
    - Load the preprocessed (128*128, values in [0,1]) CT image.pt file (shape=[N,128,128]).
    - __getitem__ returns:
      x_flat: torch.Tensor, shape=[16384], a flattened vector on the CPU
      y_cpu: torch.Tensor, shape=[measurement_dim, num_detectors], corresponding to Radon sinogram on CPU (optional)
    - Only use x_flat when training Flow. Previewing sinogram on the CPU is available with y_cpu. 
    """

    def __init__(self,
                 tensor_path='/content/drive/MyDrive/CT_dataset_128.pt',
                 measurement_dim=200):
        super().__init__()

        self.data = torch.load(tensor_path, map_location="cpu")
        self.N = self.data.shape[0]
        self.measurement_dim = measurement_dim

        # Construction of DeepInverse Tomography on CPU (only for CPU test visualization sinogram)
        angles = torch.linspace(0, np.pi, measurement_dim,
                                device="cpu", dtype=torch.float32)
        self.physics_cpu = Tomography(
            img_shape=(1, 128, 128),
            angles=angles,
            geometry="parallel",
            img_width=128,
            device="cpu"
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        return:
          x_flat: torch.Tensor, shape=[16384], flattened original image on the CPU
          y_cpu: torch.Tensor, shape=[measurement_dim, num_detectors], sinogram on CPU
        """

        x_img = self.data[idx]

        x_in = x_img.unsqueeze(0).unsqueeze(0)
        # y_full_cpu: [1, num_detectors, measurement_dim]
        y_full_cpu = self.physics_cpu(x_in) # Radon on CPU
        y_det_ang = y_full_cpu.squeeze(0).squeeze(0) # delete batch & channel -> [num_detectors, measurement_dim]
        y_cpu = y_det_ang.transpose(0, 1).contiguous() # transform -> [measurement_dim, num_detectors]

        x_flat = x_img.view(-1)

        return y_cpu, x_flat


