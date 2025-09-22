import os

from torch.utils.data import Dataset
import numpy as np
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

        self.rotation_mean = None
        self.rotation_std = None
        self.translation_mean = None
        self.translation_std = None

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
            self.poses = np.concatenate(poses_list, axis=0) 
            self.compute_pose_stats()

    def compute_pose_stats(self):
        rotations = self.poses[:, :3, :3].reshape(len(self.poses), -1)
        translations = self.poses[:, :3, 3]                             

        self.rotation_mean = rotations.mean(axis=0)
        self.rotation_std = rotations.std(axis=0) + 1e-8 

        self.translation_mean = translations.mean(axis=0)
        self.translation_std = translations.std(axis=0) + 1e-8

    def standardize_pose(self, pose):
        pose_flat = pose.reshape(-1)
        R_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        t_indices = [3, 7, 11]
    
        R = pose_flat[R_indices]
        t = pose_flat[t_indices]
    
        R = (R - self.rotation_mean) / self.rotation_std
        t = (t - self.translation_mean) / self.translation_std

        result = np.zeros(16, dtype=np.float32)
        result[R_indices] = R
        result[t_indices] = t
        result[12:] = [0, 0, 0, 1]
    
        return result
 
    def __getitem__(self, index):
        image = self.images[index]
        pose = self.poses[index]  # shape: (4, 4)
        gt_pose = pose.copy()

        if self.transform:
            image = self._to_pil_image(image)
            image = self.transform(image)

        standardized_pose = self.standardize_pose(pose)
        
        latent_size = self.latent_size
        pose_tensor = np.zeros((4, latent_size, latent_size), dtype=np.float32)

        quadrant_size = latent_size // 2
        for c in range(4):
            for i in range(2):
                for j in range(2):
                    val = standardized_pose[c * 4 + i * 2 + j]
                    pose_tensor[c, i*quadrant_size:(i+1)*quadrant_size,
                                   j*quadrant_size:(j+1)*quadrant_size] = val
        
        noise = self._make_gaussian_noise(0, 1, pose_tensor.shape)
        gaussian_pose = pose_tensor + noise
        gaussian_pose = torch.tensor(gaussian_pose, dtype=torch.float32)
        return image, gaussian_pose, gt_pose, torch.Tensor(noise)

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