import torch
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, tensor_path='/content/drive/MyDrive/CT_dataset_128.pt', measurement_dim=200, seed=42):
        self.data = torch.load(tensor_path)  # [N, 128, 128]
        self.N = self.data.shape[0]
        self.input_dim = 128 * 128
        self.measurement_dim = measurement_dim

        torch.manual_seed(seed)
        self.A = torch.randn(self.measurement_dim, self.input_dim)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.data[idx].view(-1)  # 展平为 [16384]
        y = torch.matmul(self.A, x)  # A·x
        return y, x
