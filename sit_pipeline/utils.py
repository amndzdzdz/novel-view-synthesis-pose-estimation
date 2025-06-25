import torch
import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error

def get_pose_radius(pose):
    translation_vector = pose[:3, 3] #shape [1, 32, 32]
    radius = torch.norm(translation_vector)
    
    return radius

def calculate_psnr(image, prediction):
    mse = np.mean((image - prediction) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0  # Assuming 8-bit images; change if your pixel values differ
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def evaluate_prediction(ground_truth, prediction):
    ground_truth = ground_truth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    ground_truth = ground_truth.transpose(1, 2, 0)
    prediction = prediction.transpose(1, 2, 0)

    if ground_truth.max() <= 1.0:
        ground_truth = ground_truth * 255.0
    if prediction.max() <= 1.0:
        prediction = prediction * 255.0

    ground_truth = np.clip(ground_truth, 0, 255).astype(np.float32)
    prediction = np.clip(prediction, 0, 255).astype(np.float32)
    
    psnr = calculate_psnr(ground_truth, prediction)
    mse = mean_squared_error(ground_truth.flatten(), prediction.flatten())
    
    return round(float(psnr)), round(float(mse))