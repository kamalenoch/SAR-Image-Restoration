import torch
import numpy as np
from torch.utils.data import Dataset

class SARSelfSupervised(Dataset):
    def __init__(self, images, patch=64):
        self.images = images
        self.patch = patch

    def __len__(self):
        return len(self.images) * 10

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        h, w = img.shape
        x = np.random.randint(0, h - self.patch)
        y = np.random.randint(0, w - self.patch)

        patch = img[x:x+self.patch, y:y+self.patch].copy()
        noisy = patch.copy()

        mask = np.random.rand(*patch.shape) < 0.1
        noisy[mask] = np.random.rand()

        return (
            torch.tensor(noisy).unsqueeze(0),
            torch.tensor(patch).unsqueeze(0)
        )
