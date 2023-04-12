
import cv2
import numpy as np, shutil
import random
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101

def clear_pretraining_progress():
    # Delete the pretraining_progress folder
    folder_name = "pretraining_progress"

    # Check if the folder exists and delete it if it does
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    # Recreate the folder
    os.makedirs(folder_name)
    return

def convert_mp4_to_tensor(mp4_file_path, height=224, width=224):
    '''
    mp4_file_path -> (num_frames, height, width, channels)
    '''
    # Open the MP4 video file using OpenCV
    cap = cv2.VideoCapture(mp4_file_path)

    # Initialize a list to store the tensor representations of each frame
    tensor_list = []

    # Loop over each frame of the video
    while cap.isOpened():
        # Read in the next frame of the video
        ret, frame = cap.read()

        # If there are no more frames to read, break out of the loop
        if not ret:
            break

        # Convert the frame to a tensor representation using NumPy
        tensor = np.array(frame, dtype=np.float32) / 255.0
        tensor_list.append(tensor)

    # Convert the list of tensors into a NumPy array with shape (num_frames, height, width, channels)
    tensor_array = np.array(tensor_list)

    # Release the OpenCV video capture object
    cap.release()
    video_tensor = apply_pos_enc(tensor_array)


    # Resize video tensor to have height and width be 224 x 224
    # Convert the video tensor to a list of individual frames (PIL Images)
    frames = [transforms.ToPILImage()(frame) for frame in video_tensor]

    # Define the resize transformation
    resize_transform = transforms.Resize((height, width))

    # Apply the resize transformation to each frame
    resized_frames = [resize_transform(frame) for frame in frames]

    # Convert the resized frames back to tensors and stack them to form the resized video tensor
    resized_video_tensor = torch.stack([transforms.ToTensor()(frame) for frame in resized_frames])

    return resized_video_tensor

def get_segmentation_mask(segmented_frame):
    # Load the pre-trained model
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Move the model and image to the same device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # Add the batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0)

    # Get the output from the model
    with torch.no_grad():
        output = model(image_tensor)

    # Get the segmentation mask
    segmentation_mask = output['out'].argmax(dim=1)

    return segmentation_mask

def get_first_frame_half_segmented(video_tensor):


    # Get the segmentation mask

    first_frame_tensor = video_tensor[0:1, :, :, :]

    segmentation_mask = get_segmentation_mask(first_frame_tensor)

    # Convert the segmentation mask to the same format as the input image tensor
    segmentation_mask = segmentation_mask.permute(0, 2, 1).unsqueeze(3)

    # Split the input image and the segmentation mask in half
    height = first_frame_tensor.shape[1]
    top_half_original = first_frame_tensor[:, :height // 2, :, :]
    bottom_half_segmentation = segmentation_mask[:, height // 2:, :, :]
    
    # Concatenate the top half of the original image with the bottom half of the segmentation mask
    combined_image_tensor = torch.cat((top_half_original, bottom_half_segmentation), dim=1)

    return combined_image_tensor
