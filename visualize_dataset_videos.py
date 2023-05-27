from video_mae_code.dataset_factory import get_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import Video
import os
import torchvision.transforms as transforms

### 
dataset_name = "SSV2"
index = 99
###

# get the dataset and object tensor
dataset = get_dataset(dataset_name, os.path.join(os.path.expanduser("~"), "Datasets"), "video")
object_tensor = dataset[index][0]  # torch.Size([1, 3, 16, 224, 224]

print(object_tensor.shape)

# transpose to get correct dimensions: [frames, height, width, channels]
object_tensor = object_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

# normalize tensor to [0,1]
object_tensor = (object_tensor - object_tensor.min()) / (object_tensor.max() - object_tensor.min())

# prepare frames
fig, ax = plt.subplots()
frames = [[plt.imshow(frame, animated=True)] for frame in object_tensor]

# Hide axes and remove borders
plt.axis('off')
plt.grid(False)

ani = ArtistAnimation(fig, frames, interval=50, blit=True)

# Include index in the video name
video_name = f"visualized_video_from_{dataset_name}_{index}.mp4"
ani.save(video_name, fps=4)

# Close the figure so it doesn't get displayed
plt.close(fig)
