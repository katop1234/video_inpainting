
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

from typing import Iterable
import util.lr_sched as lr_sched
import util.misc as misc
import util.video_vit as video_vit
import torch
import numpy as np
from dataset_factory import get_imagenet_val_dataloader, get_linprobe_train_dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import wandb

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
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
        "mask_ratio_image", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio_video", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, ((samples, _), accum_iter) in enumerate(
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

        #Added back
        if len(samples.shape) == 4: # NOTE this is only when using original video inpainting dataset_train has shape (N, C, H, W)
            samples = samples.unsqueeze(2) # add the num_frames dimension
        
        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, _, _ = model(
                samples,
                mask_ratio_image=args.mask_ratio_image, 
                mask_ratio_video=args.mask_ratio_video
            )

        loss_value = loss.item()
        assert not np.isnan(loss_value), 'loss is nan'

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
        metric_logger.update(mask_ratio_image=args.mask_ratio_image)
        metric_logger.update(mask_ratio_video=args.mask_ratio_video)

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

    # Imagenet Probing
    imagenet_probing_freq = 5
    num_epochs = 90 # from MAE 
    num_samples = 512
    if epoch % imagenet_probing_freq == 0 and misc.is_main_process():
        ### Imagenet probing training
        probe = torch.nn.Sequential(torch.nn.BatchNorm1d(49, affine=False, eps=1e-6), probe).to(device)  # wrap the linear layer with BatchNorm1d
        torch.nn.init.trunc_normal_(probe.weight, std=0.01)
        probe_optimizer = probe_optimizer = video_vit.LARS(probe.parameters(), lr=1.5e-4, weight_decay=0.05).to(device) # params from MAE
        imagenet_train_dataset = get_linprobe_train_dataset()
        
        for param in model.module.parameters():
            param.requires_grad = False

        for param in probe.parameters():
            param.requires_grad = True

        for epoch in range(num_epochs):
            indices = list(range(len(imagenet_train_dataset)))
            np.random.shuffle(indices)
            sampler = SubsetRandomSampler(indices[:num_samples])
            data_loader = DataLoader(imagenet_train_dataset, batch_size=64, sampler=sampler, num_workers=14)

            print("Linear probing Epoch: {}".format(epoch))
            for samples, labels in data_loader:
                samples = samples.permute(0, 2, 1, 3, 4).to(device)  # Now samples shape is (B, C, T, H, W)
                labels = labels.to(device)
                        
                probe_optimizer.zero_grad()

                with torch.no_grad():
                    latents = model(samples, imagenet_probing=True)
                output = probe(latents)
                output = output.mean(dim=1)  # Take the mean across the patch dimension
                loss = torch.nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                probe_optimizer.step()

        ### Imagenet evaluation
        imagenet_val_dataloader = get_imagenet_val_dataloader()
            
        correct = 0
        total = 0

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients
            for samples, labels in imagenet_val_dataloader:
                samples = samples.permute(0, 2, 1, 3, 4).to(device)  # Now samples shape is (B, C, T, H, W)
                labels = labels.to(device)

                latents = model(samples, imagenet_probing=True)
                output = probe(latents)
                output = output.mean(dim=1)  # Take the mean across the patch dimension

                _, predicted = torch.max(output.data, 1)  # Get the predicted classes
                total += labels.size(0)  # Increment the total count
                correct += (predicted == labels).sum().item()  # Increment the correct count

        accuracy = 100 * correct / total
        print(f'Accuracy on the validation images: {accuracy}%')
        
        for param in model.module.parameters():
            param.requires_grad = True

        for param in probe.parameters():
            param.requires_grad = False
            
        model.train()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
    train_stats["imagenet_val_accuracy"] = accuracy if epoch % imagenet_probing_freq == 0 else None
    return train_stats
