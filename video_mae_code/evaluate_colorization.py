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
    
    mse = torch.mean((target - ours)**2)
    return {'mse': mse}


def run_evaluation_method(store_path):
    mse_list = []
    
    for filename in os.listdir(store_path):
        video_path = os.path.join(store_path, filename)
        cap = cv2.VideoCapture(video_path)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, c = frame.shape
            # Assuming the ground truth is in the bottom right quadrant and prediction is in the bottom left quadrant
            
            ours = frame[h//2:, w//2:]

            metric = calculate_metric(ours, target)
            mse_list.append(metric['mse'])
        
        cap.release()
        cv2.destroyAllWindows()
    
    single_mean_2x2 = np.mean(mse_list)
    
    # Assuming you have a similar procedure for single_mean_image but didn't provide it here
    # You can extend it similarly
    single_mean_image = 0  # Placeholder. You'll replace this with the appropriate code.

    return single_mean_2x2, single_mean_image
