import torch
import wandb
import os
import numpy as np
import models_mae
import cv2
from PIL import Image
from util.decoder import utils
import random


def save_frames_as_mp4(frames: torch.Tensor, file_name: str):
    '''
    "The input tensor should have the shape: (N, C, T, H, W)"
    '''
    frames_np = frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames_np = np.clip(frames_np, 0, 255)
    frames_uint8 = frames_np.astype(np.uint8)
    num_frames, height, width, _ = frames_uint8.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = cv2.cvtColor(frames_uint8[i], cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    return frames_uint8


def save_test_output(output, name):
    if output.shape == (1, 3, 16, 224, 224):
        return save_frames_as_mp4(output, name)
    elif output.shape == (224, 224, 3):
        return Image.fromarray(output).save(name)
    raise NotImplementedError


def get_random_file(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    random_file = random.choice(files)
    return os.path.join(data_dir, random_file)


def video_to_tensor(video_path, target_size=(224, 224), num_frames=16):
    '''
    Converts a given video mp4 file to a PyTorch tensor
    NOT normalized
    '''
    # Read the video
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the sampling rate to get the target number of frames
    sampling_rate = total_frames // num_frames

    # Initialize an empty list to store the resized frames
    resized_frames = []

    frame_count = 0
    sampled_count = 0

    # Loop through the video frames
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % sampling_rate == 0:
            # Resize the frame
            resized_frame = cv2.resize(frame, target_size)

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Append the resized frame to the list
            resized_frames.append(rgb_frame)
            sampled_count += 1

            if sampled_count >= num_frames:
                break

        frame_count += 1

    # Release the video capture object
    video.release()

    # Convert the list of frames to a PyTorch tensor
    resized_frames = np.array(resized_frames)
    video_tensor = torch.tensor(resized_frames, dtype=torch.float32)

    # Rearrange the tensor dimensions to (batch, channel, time, height, width)
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)

    assert video_tensor.shape == (1, 3, 16, 224, 224)

    return video_tensor


def check_folder_equality(str1, str2):
    n = len(str2)
    return str1[-n:] == str2


def image_to_tensor(image_path, target_shape=(1, 3, 1, 224, 224)):
    '''
    Returns a tensor of shape (1, 3, 1, 224, 224) from an image
    NOT normalized
    '''
    img = Image.open(image_path)
    img = img.resize((target_shape[-1], target_shape[-2]), Image.ANTIALIAS)  # Resize to (width, height)
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Convert to a tensor and rearrange dimensions
    img = img.unsqueeze(1)  # Add the num_frames dimension
    img = img.unsqueeze(0)  # Add the batch_size dimension

    assert img.shape == target_shape
    return img.float()


def uint8_to_normalized(tensor):
    """
    Convert a uint8 tensor to a float tensor and normalize the values.
    tensor: PyTorch tensor, the uint8 tensor to convert
    """
    return utils.tensor_normalize(tensor)


def normalized_to_uint8(tensor):
    output = utils.revert_tensor_normalize(tensor)
    output = torch.round(output)
    return output


def get_test_model_input(file: str = None, data_dir: str = None):
    # First check if direct file exists
    if file:
        if file.endswith(".mp4"):
            video_tensor = video_to_tensor(file)
            video_tensor = uint8_to_normalized(video_tensor)
            return video_tensor.cuda()
        elif file.endswith(".png"):
            image_tensor = image_to_tensor(file, (1, 3, 1, 224, 224))
            image_tensor = uint8_to_normalized(image_tensor)
            return image_tensor.cuda()
        raise NotImplementedError

    # TODO also feed in "test_cases/final_spatiotemporal_videos/"
    if check_folder_equality(data_dir, "test_cases/final_temporal_videos/"):
        random_mp4 = get_random_file(data_dir)
        return get_test_model_input(file=random_mp4)

    elif check_folder_equality(data_dir, "test_cases/visual_prompting_images/"):
        random_png = get_random_file(data_dir)
        return get_test_model_input(file=random_png)

    raise NotImplementedError


def spatial_sample_test_video(test_model_input):
    spatial_idx = 1
    test_model_input = utils.spatial_sampling(
        test_model_input.squeeze(0),
        spatial_idx=spatial_idx,
        min_scale=256,
        max_scale=256,
        crop_size=224,
        random_horizontal_flip=False
    )
    test_model_input = test_model_input.unsqueeze(0)
    return test_model_input  # shape (1, 3, 16, 224, 224)


def reconstruct(mask, ground_truth, test_model_output):
    expanded_mask = mask.unsqueeze(-1).expand_as(ground_truth)
    result = torch.where(expanded_mask == 1, test_model_output, ground_truth)
    return result


@torch.no_grad()
def visualize_prompting(model, input_image_viz_dir, input_video_viz_dir):
    model.eval()
    visualize_image_prompting(model, input_video_viz_dir)
    visualize_video_prompting(model, input_image_viz_dir)
    model.train()

@torch.no_grad()
def visualize_image_prompting(model, input_image_viz_dir):
    ### Test on images
    for i, img_file in enumerate(os.listdir(input_image_viz_dir)):
        img_file = os.path.join(input_image_viz_dir, img_file)
        test_model_input = get_test_model_input(file=img_file)
        test_model_input = test_model_input.cuda()

        with torch.no_grad():
            _, test_model_output, mask = model(test_model_input, test_image=True)

        if type(model) is torch.nn.parallel.DistributedDataParallel:
            patchified_gt = model.module.patchify(test_model_input)
            reconstructed_output = reconstruct(mask, patchified_gt, test_model_output)
            test_model_output = model.module.unpatchify(reconstructed_output)
        elif type(model) is models_mae.MaskedAutoencoderViT:
            patchified_gt = model.patchify(test_model_input)
            reconstructed_output = reconstruct(mask, patchified_gt, test_model_output)
            test_model_output = model.unpatchify(reconstructed_output)
        else:
            raise NotImplementedError("Something's funky")

        test_model_output = normalized_to_uint8(test_model_output)
        denormalized_img = test_model_output.squeeze(0).permute(1, 2, 3, 0).squeeze(0)  # (224, 224, 3)
        image_array = (denormalized_img.cpu().numpy()).astype(np.uint8)
        output_img_name = 'test_model_output_img' + str(i) + '.png'
        image = wandb.Image(image_array)
        wandb.log({output_img_name: image})


@torch.no_grad()
def visualize_video_prompting(model, input_video_viz_dir="test_cases/final_temporal_videos/"):
    test_model_input = get_test_model_input(data_dir=input_video_viz_dir)
    test_model_input = spatial_sample_test_video(test_model_input)

    with torch.no_grad():
        # TODO change test_temporal to True later when it works
        _, test_model_output, _ = model(test_model_input)

    if type(model) is torch.nn.parallel.DistributedDataParallel:
        test_model_output = model.module.unpatchify(test_model_output)
    elif type(model) is models_mae.MaskedAutoencoderViT:
        test_model_output = model.unpatchify(test_model_output)
    else:
        raise NotImplementedError("Something's funky")

    test_model_output = normalized_to_uint8(test_model_output)
    test_model_output_np = test_model_output.squeeze(0).permute(1, 0, 3, 2).cpu().numpy()

    wandb_video_object = wandb.Video(
        data_or_path=test_model_output_np,
        fps=30
    )
    wandb.log({"video": wandb_video_object})
