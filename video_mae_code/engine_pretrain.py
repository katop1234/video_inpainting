# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math, cv2
import util
from typing import Iterable
import random
import numpy as np
from PIL import Image

import util.lr_sched as lr_sched
import util.misc as misc
import os
import torch

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    accum_iter_determined_from_batch_size,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    fp32=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = accum_iter_determined_from_batch_size # calculated from batch_size

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):  
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape # r is number of repeated variations

            samples = samples.reshape(b * r, c, t, h, w) # flatten the repeated variations to batches
        
        if len(samples.shape) == 4: # TODO this is only when using original video inpainting dataset_train has shape (N, C, H, W)
            samples = samples.unsqueeze(2) # add the num_frames dimension

        with torch.cuda.amp.autocast(enabled=not fp32): # DEBUG follow the dims of samples, it should throw an error
            loss, _, _ = model(
                samples,
                mask_ratio_image=0.75, # default .75 # TODO allow to feed this in as an argument
                mask_ratio_video=args.mask_ratio # fixed hyperparameter at 0.9
            )

        loss_value = loss.item()

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0, # updates grad every accum_iter
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad() # zeroes out grad every accum iter

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=args.mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
        
        if data_iter_step % 1000 == 0:
            print("Epoch: {}, Iter: {}, Loss: {}".format(epoch, data_iter_step, loss_value_reduce))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_random_file(data_dir):
    # List all files in the data directory
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    # Choose a random file
    random_file = random.choice(files)
    
    # Return the random file's path
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

def uint8_to_normalized(tensor, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
    """
    Convert a uint8 tensor to a float tensor and normalize the values.
    tensor: PyTorch tensor, the uint8 tensor to convert
    """
    # NOTE try to use a fixed mean and std all throughout training
    return util.decoder.utils.tensor_normalize(tensor, mean, std)

def normalized_to_uint8(tensor, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
    if type(mean) is tuple:
        mean = list(mean)
    if type(std) is tuple:
        std = list(std)
        
    return util.decoder.utils.revert_tensor_normalize(tensor, mean, std)

def get_test_model_input(file:str=None, data_dir:str=None):
        
    # First check if direct file exists
    if file:
        if file.endswith(".mp4"):
            video_tensor = video_to_tensor(file)
            video_tensor = uint8_to_normalized(video_tensor)
            return video_tensor.cuda()
        elif file.endswith(".png"):
            image_tensor = image_to_tensor(file, (1, 3, 1, 224, 224)).cuda()
            image_tensor = uint8_to_normalized(image_tensor)
            return image_tensor
        raise NotImplementedError
            
    # If not, go to data_dir and get a random file
    # TODO also feed in "test_cases/final_spatiotemporal_videos/"
    if data_dir == "test_cases/final_temporal_videos/":
        random_mp4 = get_random_file(data_dir)
        video_tensor = video_to_tensor(random_mp4)
        video_tensor = uint8_to_normalized(video_tensor)
        return video_tensor.cuda()
    elif data_dir == "test_cases/visual_prompting_images/":
        random_png = get_random_file(data_dir)
        image_tensor = image_to_tensor(random_png, (1, 3, 1, 224, 224))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
            
        image_tensor = uint8_to_normalized(image_tensor, mean, std)
        return image_tensor.cuda()
    raise NotImplementedError

def spatial_sample_test_video(test_model_input):
  
    # Deterministic spatial sampling (center crop)
    spatial_idx = 1 # applies center crop
    test_model_input = util.decoder.utils.spatial_sampling(
            test_model_input.squeeze(0),
            spatial_idx=spatial_idx,
            min_scale=256,
            max_scale=256,
            crop_size=224,
            random_horizontal_flip=False
                        )
    test_model_input = test_model_input.unsqueeze(0)
    return test_model_input # shape (1, 3, 16, 224, 224)

