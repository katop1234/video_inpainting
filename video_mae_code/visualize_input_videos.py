
from util.eval import decode_raw_prediction, get_test_model_input, spatial_sample_test_video
from dataset_factory import get_dataset
import wandb
import os
import numpy as np
import torch
import cv2
from datetime import datetime

'''
Generates sample input videos for each dataset to visualize and logs them to wandb.
Specify how many videos to generate for each dataset in the dictionary below.
'''

### Change here
dataset_counts = {
    "atari": 20,
    "SSV2": 5,
    "CrossTask": 5,
    "Objectron": 5,
    "kinetics": 5,
}
###

def visualize_input_from_dataset(dataset_name):  
    
    dataset = get_dataset(name=dataset_name, root_path="/shared/katop1234/Datasets/", ds_type="video")
    
    index = np.random.randint(0, len(dataset))
    test_model_input = dataset[index][0]

    test_model_input = test_model_input.permute(0, 2, 1, 3, 4)
    test_model_input = test_model_input.squeeze(0)
    test_model_input = 255 * (test_model_input - test_model_input.min()) / (test_model_input.max() - test_model_input.min())
    test_model_input = (test_model_input.cpu().numpy()).astype(np.uint8)

    wandb_video_object = wandb.Video(
        data_or_path=test_model_input,
        fps=4,
        format="mp4"
    )
    
    # Prepare video name with seconds and milliseconds
    folder_name = dataset_name
    current_time = datetime.now().strftime('%M%S%f')[:-3]  # Include milliseconds to differentiate videos from same dataset
    video_name = f"{folder_name}_sample_input_{current_time}.mp4"

    # Log the video name and video object with wandb
    wandb.log({video_name: wandb_video_object})

wandb.init()

for dataset in dataset_counts.keys():
    for _ in range(dataset_counts[dataset]):
        visualize_input_from_dataset(dataset)
        
