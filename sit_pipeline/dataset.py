from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import torch

class SiTDataset(Dataset):
    def __init__(self, pth=None, transform=None, image_size=128):   
        self.pth = []
        self.images = []
        self.poses = []
        self.focal_length = []
        self.transform = transform

        self.image_size = image_size                      
        self.latent_size = image_size // 8                

        self.pose_mean = None
        self.pose_std = None

        if pth is not None:
            self.pth = pth

            images_list = []
            poses_list = []

            for filename in os.listdir(pth):
                filepath = os.path.join(pth, filename)
                loaded = np.load(filepath)
                images_list.append(loaded["images"])
                poses_list.append(loaded["poses"])

            self.focal_length = loaded["focal"]
            self.images = np.concatenate(images_list, axis=0)
            self.poses = np.concatenate(poses_list, axis=0)  # shape: (N, 4, 4)

            self.compute_pose_stats()

    def compute_pose_stats(self):
        poses_flat = self.poses.reshape(len(self.poses), -1)  # shape: (N, 16)
        self.pose_mean = poses_flat.mean(axis=0)              # shape: (16,)
        self.pose_std = poses_flat.std(axis=0) + 1e-8         # avoid division by zero

    def __getitem__(self, index):
        image = self.images[index]
        pose = self.poses[index]  # shape: (4, 4)
        gt_pose = pose.copy()

        if self.transform:
            image = self._to_pil_image(image)
            image = self.transform(image)

        pose_flat = pose.reshape(-1) 
        pose_normalized = (pose_flat - self.pose_mean) / self.pose_std

        latent_size = self.latent_size
        pose_tensor = np.zeros((4, latent_size, latent_size), dtype=np.float32)

        quadrant_size = latent_size // 2
        idx = 0
        for c in range(4):
            for i in range(2):
                for j in range(2):
                    val = pose_normalized[c * 4 + i * 2 + j]
                    pose_tensor[c, i*quadrant_size:(i+1)*quadrant_size,
                                   j*quadrant_size:(j+1)*quadrant_size] = val
        
        noise = self._make_gaussian_noise(0, 1, pose_tensor.shape)
        gaussian_pose = pose_tensor + noise
        gaussian_pose = torch.tensor(gaussian_pose, dtype=torch.float32)

        return image, gaussian_pose, gt_pose, noise

    def _to_pil_image(self, image):
        return transforms.ToPILImage()(image)

    def _make_gaussian_noise(self, mean: float, std: float, size: tuple):
        noise = np.random.normal(mean, std, size).astype(np.float32)
        return noise

    def __len__(self):
        if self.images is not None:
            return self.images.shape[0]
        return 0

    def save(self):
        np.savez_compressed(self.pth, images=self.images, poses=self.poses, focal=self.focal_length)
