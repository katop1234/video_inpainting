
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
import torch
import numpy as np
import logging
import sys
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

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger()
        logger.info("engine_pretrain.py start of train_one_epoch")
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            logger.info('engine_pretrain.py in lr check')
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )
            logger.info('engine_pretrain.py after lr check')

        logger.info('engine_pretrain.py before moving samples')
        samples = samples.to(device, non_blocking=True)
        logger.info('engine_pretrain.py after moving samples')
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape # r is number of repeated variations

            samples = samples.reshape(b * r, c, t, h, w) # flatten the repeated variations to batches

        #Added back
        if len(samples.shape) == 4: # NOTE this is only when using original video inpainting dataset_train has shape (N, C, H, W)
            samples = samples.unsqueeze(2) # add the num_frames dimension
        
        logger.info('engine_pretrain.py before model forward')
        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, _, _ = model(
                samples,
                mask_ratio_image=args.mask_ratio_image, 
                mask_ratio_video=args.mask_ratio_video
            )
        logger.info('engine_pretrain.py after model forward')

        loss_value = loss.item()
        assert not np.isnan(loss_value), 'loss is nan'

        logger.info('engine_pretrain.py before loss scaler')
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0, # updates grad every accum_iter
            clip_grad=args.clip_grad,
        )
        logger.info('engine_pretrain.py after loss scaler')

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad() # zeroes out grad every accum iter

        logger.info('engine_pretrain.py before synchronize')
        torch.cuda.synchronize()
        logger.info('engine_pretrain.py after synchronize')

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


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
