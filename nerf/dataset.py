"""
This code is taken from my supervisor Ulrich Prestel, PHD student at LMU Munich.
https://github.com/uprestel/AutoNeRF
"""
import numpy as np
from torch.utils.data import Dataset

class NeRF_Dataset(Dataset):
    def __init__(self, pth=None):
        self.pth = None
        self.images = None
        self.poses = None
        self.focal_length = None
        
        if pth is not None:
            self.pth = pth
            loaded = np.load(self.pth)

            self.images = loaded["images"]
            self.poses = loaded["poses"]
            self.focal_length = loaded["focal"]

    def __getitem__(self, index):
        image = self.images[index]
        pose = self.poses[index]

        return image, pose

    def __len__(self):
        if self.images is not None:
            return self.images.shape[0]
        return 0

    def save(self):
        np.savez_compressed(self.pth,
                            images = self.images,
                            poses = self.poses,
                            focal = self.focal_length
                            )