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

    print("ENTERING THE FOR LOOP NEW CODE")

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):  
        print("samples acquired, data_iter_step", data_iter_step)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape # r is number of repeated variations

            samples = samples.reshape(b * r, c, t, h, w) # flatten the repeated variations to batches

        print("samples shape", samples.shape)
        print("RUNNING FORWARD PASS ON THE MODEL new code \n")

        with torch.cuda.amp.autocast(enabled=True):
            loss, _, _ = model(
                samples,
                mask_ratio=args.mask_ratio,
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
    
    # Test data once the epoch is over
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder name with the current date and time
    folder_name = f"pretraining_progress/progress_{current_datetime}"

    # Create the folder if it doesn't already exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
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

    def get_test_model_input_nomasking(data_dir):
        # keep the top half of the video, and the bottom half of the first frame. mask everything else.
        file_path = get_random_file(data_dir)
        video_tensor = video_to_tensor(file_path)
        return video_tensor

    def get_test_model_input(data_dir="test_cases/final_temporal_videos/"):
        # TODO also feed in "test_cases/final_temporal_videos/"
        tensor_video = get_test_model_input_nomasking(data_dir)
        return tensor_video
    
    def output_to_mp4(output, output_file_path, output_shape):
        '''
        Make sure that the output of the decoder makes sense and we can convert it to a video file.
        Go through it line by line if needed. Make sure you sue the original unmasked patches,
         and the reconstructions for the masked ones.
        '''
        raise NotImplementedError
        # # Rearrange patches to form a video tensor
        # output_video_tensor = output.view(*output_shape)
        # output_video_tensor = output_video_tensor.permute(0, 2, 3, 4, 1)  # Rearrange dimensions to (batch, time, height, width, channel)
        # output_video_tensor = output_video_tensor.squeeze(0)  # Remove the batch dimension
        
        # # Convert the tensor back to the range [0, 255] and change its data type to uint8
        # output_video_tensor = (output_video_tensor * 255).clamp(0, 255).byte().cpu().numpy()
        
        # # Write the frames to a video file
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(output_file_path, fourcc, 30, (224, 224))
        
        # for i in range(output_video_tensor.shape[0]):
        #     frame = output_video_tensor[i]
        #     bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     video_writer.write(bgr_frame)
        
        # video_writer.release()
    
    # Starting evaluation
    model.eval()
    test_model_input = get_test_model_input()

    with torch.no_grad():
        _, test_model_output, _ = model(test_model_input, test_temporal=True)

    output_file_path = os.path.join(folder_name, f"output_{data_iter_step}.mp4")
    output_shape = (1, 3, 16, 224, 224)

    output_to_mp4(test_model_output, output_file_path, output_shape)
    model.train()

    # End evaluation

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
