from video_mae_code.dataset_factory import get_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import Video
import os
import torchvision.transforms as transforms

### 
dataset = "atari"
index = 1061
###

# get the dataset and object tensor
dataset = get_dataset(dataset, os.path.join(os.path.expanduser("~"), "Datasets"), "video")
object_tensor = dataset[index][0]  # torch.Size([1, 3, 16, 224, 224]

print(object_tensor.shape)

# transpose to get correct dimensions: [frames, height, width, channels]
object_tensor = object_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

# normalize tensor to [0,1]
object_tensor = (object_tensor - object_tensor.min()) / (object_tensor.max() - object_tensor.min())

# prepare frames
fig, ax = plt.subplots()
frames = [[plt.imshow(frame, animated=True)] for frame in object_tensor]

ani = ArtistAnimation(fig, frames, interval=50, blit=True)
ani.save("video.mp4", fps=4)

# Close the figure so it doesn't get displayed
plt.close(fig)

# Display the video
Video("video.mp4")
