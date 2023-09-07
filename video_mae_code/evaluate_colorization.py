import cv2
import numpy as np
import torch
import os

# Assuming these are the ImageNet mean and std
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def calculate_metric(ours, target):
    # Convert tensors to numpy arrays if they are tensors
    assert ours.shape == target.shape
    if isinstance(ours, torch.Tensor):
        ours = ours.float() # Convert to float
        ours /= 255.       # Normalize)
    if isinstance(target, torch.Tensor):
        target = target.float() # Convert to float
        target /= 255.         # Normalize
    
    mse = torch.mean((target - ours)**2).item()
    return {'mse': mse}