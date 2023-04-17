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
from PIL import Image
from typing import Iterable
import itertools, random, os
import numpy as np, shutil

import util.lr_sched as lr_sched
import util.misc as misc
import os
from datetime import datetime
import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from kinetics_and_images import KineticsAndCVF

# Create the folder name with the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"pretraining_progress/progress_{current_datetime}"

# Create the folder if it doesn't already exist
if not os.path.exists(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError: # Multiple GPUs might be doing this simulatenously
       pass

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

    accum_iter = accum_iter_determined_from_batch_size

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

        with torch.cuda.amp.autocast(enabled=True):
            loss, _, _ = model(
                samples,
                mask_ratio_image=args.mask_ratio, # default .75
                mask_ratio_video=0.9 # TODO hyperparameter

            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

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
    
    return video_tensor

def load_image_as_tensor(image_path, target_shape=(3, 1, 224, 224)):
    img = Image.open(image_path)
    img = img.resize((target_shape[-1], target_shape[-2]), Image.ANTIALIAS)  # Resize to (width, height)
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Convert to a tensor and rearrange dimensions
    img = img.unsqueeze(1)  # Add the num_frames dimension
    img = img.unsqueeze(0)  # Add the batch_size dimension
    return img

def get_test_model_input_nomasking(data_dir):
    # keep the top half of the video, and the bottom half of the first frame. mask everything else.
    file_path = get_random_file(data_dir)
    video_tensor = video_to_tensor(file_path)
    return video_tensor

png_files = []

for root, _, files in os.walk("/shared/amir/dataset/arxiv_resized_train_val_split/train/"):
    for file in files:
        if file.lower().endswith('.png'):
            png_files.append(os.path.join(root, file))

if not png_files:
    raise FileNotFoundError("No PNG files found in the folder")

def get_test_model_input(data_dir="test_cases/final_temporal_videos/"):
    # TODO also feed in "test_cases/final_spatiotemporal_videos/"
    
    if data_dir == "test_cases/final_temporal_videos/":
        tensor_video = get_test_model_input_nomasking(data_dir)
        return tensor_video
    elif data_dir == "/shared/amir/dataset/arxiv_resized_train_val_split/train/":
        random_png = random.choice(png_files)
        tensor_image = load_image_as_tensor(random_png, (3, 1, 224, 224))
        return tensor_image
    else:
        raise NotImplementedError
