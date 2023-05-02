# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import torchvision.transforms as transforms
from torchvision import datasets
import datetime
# from kinetics_and_images import save_frames_as_mp4
import wandb
import json
import os
import time
from util.eval import visualize_prompting
import util.decoder.constants as constants
# import util.env

import util.misc as misc

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
import models_mae
from engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.tensorboard import SummaryWriter

import util.decoder.utils as utils

###
test_image = False
test_video = False
###

assert not (test_image and test_video), "Can't test both image and video"

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )

    parser.add_argument("--epochs", default=4000, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="*We calculate this automatically to match effective batch size*. Accumulate gradient iterations (for increasing the effective batch size under memory constraints).",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--mask_ratio_video",
        default=0.9,
        type=float,
        help="Masking ratio (percentage of removed patches). 0.9 for video, and 0.75 for images",
    )
    
    parser.add_argument(
        "--mask_ratio_image",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches). 0.9 for video, and 0.75 for images",
    )

    parser.add_argument(
        "--norm_pix_loss",  # keep this false because it normalizes within N(0, 1) in a patch
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )

    parser.add_argument(
        "--blr",
        type=float,
        default=1e-5,  # NOTE was 1.6e-3 on the mae st code # NOTE was 1e-4 from amir
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=15, metavar="N", help="epochs to warmup LR"  # NOTE was 5 on mae st
    )

    parser.add_argument(
        "--path_to_data_dir",
        default="",
        help="KINETICS_DIR or IMAGES DIR. I hardcoded this so don't worry about it.",
    )

    parser.add_argument(
        "--image_prompts_dir",
        default="/shared/katop1234/video_inpainting/video_inpainting/test_cases/visual_prompting_images/",
        help="Image folder containing visualization examples.",
    )

    parser.add_argument(
        "--video_prompts_dir",
        default="/shared/katop1234/video_inpainting/video_inpainting/test_cases/final_temporal_videos/",
        help="Folder containing video visualization examples.",
    )


    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )

    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument("--num_workers", default=14, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=8, type=int, help="number of distributed processes"
    )

    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")

    # Video related configs
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)  # NOTE amir said to make this 8 when doing only images
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=1, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=float("inf"),  # NOTE changed this from 0.02 to inf
    )
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--trunc_init",
        action="store_true",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.set_defaults(fp32=False)
    parser.add_argument(
        "--jitter_scales_relative",
        default=[0.5, 1.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--jitter_aspect_relative",
        default=[0.75, 1.3333],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--pred_t_dim",
        type=int,
        default=16,
    )
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    return parser

def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # I added this line because it's needed in new torch update
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # WARNING uncomment this to get both Kinetics and CVF dataset
    # dataset_train = KineticsAndCVF( # Custom Dataset
    #     mode="pretrain",
    #     path_to_csv="/shared/katop1234/video_inpainting/video_inpainting/kinetics_videos.csv", # This is the path to names of all kinetics files
    #     path_to_data_dir="/shared/group/kinetics/train_256/", # video
    #     path_to_image_data_dir="/shared/amir/dataset/arxiv_resized_train_val_split/train/", # images
    #     sampling_rate=args.sampling_rate, # 4 by default
    #     num_frames=args.num_frames, # 16 by default
    #     train_jitter_scales=(256, 320),
    #     repeat_aug=args.repeat_aug,
    #     jitter_aspect_relative=args.jitter_aspect_relative,
    #     jitter_scales_relative=args.jitter_scales_relative,
    # )
    # print("got dataloader")

    # class CustomDistributedSampler(torch.utils.data.DistributedSampler):
    #     def __init__(self, dataset, batch_size=args.batch_size, num_replicas=None, rank=None):
    #         # Call the base class constructor
    #         super().__init__(dataset, 
    #                          num_replicas=num_replicas, 
    #                          rank=rank, 
    #                          shuffle=False) # No need to shuffle, I randomly sample anyway

    #         self.dataset = dataset
    #         self.num_images = dataset.num_images
    #         self.num_videos = dataset.num_videos
    #         self.batch_size = batch_size

    #         self.image_indices = dataset.image_indices
    #         self.video_indices = dataset.video_indices

    #         self.prob_choose_image = 1 # WARNING change this to 0.5 later

    #     def __len__(self):
    #         if self.prob_choose_image == 1:
    #             return self.num_images
    #         elif self.prob_choose_image == 0:
    #             return self.num_videos
    #         return len(self.dataset)

    #     def __iter__(self):
    #         custom_indices = []

    #         num_elements_in_epoch = self.dataset.num_images # size of CVF dataset NOTE can change later

    #         if test_image or test_video:
    #             num_elements_in_epoch = 1
    #             if test_image:
    #                 self.prob_choose_image = 1 
    #             else:
    #                 self.prob_choose_image = 0

    #         while len(custom_indices) < num_elements_in_epoch:
    #             if random.random() < self.prob_choose_image: # Choose image
    #                 # append batch_size number of images
    #                 random_image_indices = random.sample(self.image_indices, self.batch_size)
    #                 custom_indices.extend(random_image_indices)
    #             else: # Choose video
    #                 # append batch_size number of videos
    #                 random_video_indices = random.sample(self.video_indices, self.batch_size)
    #                 custom_indices.extend(random_video_indices)

    #         self.custom_indices = custom_indices

    #         # Return the modified indices as an iterator
    #         return iter(self.custom_indices)

    # simple augmentation
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=constants.mean, std=constants.std)])

    dataset_train = datasets.ImageFolder("/home/amir/Datasets/arxiv_resized_train_val_split/train/",
                                         transform=transforms_train)

    if True:
        num_tasks = misc.get_world_size()  # 8 gpus
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )  # Original

    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # WARNING uncommented this to replicate original results exactly
    # sampler_train = CustomDistributedSampler(
    #         dataset_train, batch_size=args.batch_size, num_replicas=num_tasks, rank=global_rank
    #     ) # My own for videos + images 
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    '''
    lr = blr * batch_size * accum_iters * num_gpus
    
    for facebook they used
    lr = 1.6e-3 * 512 * 1 * 128
    
    Therefore we use
    lr = 1.6e-3 * B * A * 8 = 1.6e-3 * 512 * 1 * 128

    So B * A = 512 * 128 / 8 = 8192
    '''

    # WARNING use this to match MAE ST 
    # assert 8192 % args.batch_size == 0
    # accum_iter_determined_from_batch_size = 8192 // args.batch_size

    # WARNING I changed this to match amir's inpainting code
    accum_iter_determined_from_batch_size = 64 // args.batch_size
    args.accum_iter = accum_iter_determined_from_batch_size

    print("Batch size is", args.batch_size)
    print("Accumulate iterations is", args.accum_iter)
    print("Num GPUs is", misc.get_world_size())

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[args.model](
        **vars(args),
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # (8192) * 8
    eff_batch_size = args.batch_size * accum_iter_determined_from_batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if not (test_image or test_video):
        # From yossi's code for wandb
        wandb_config = vars(args)
        base_lr = (args.lr * 256 / eff_batch_size)
        wandb_config['base_lr'] = base_lr

        if misc.is_main_process():
            wandb.init(
                project="video_inpainting2",
                resume=False,
                config=wandb_config)
    # From yossi's code for wandb

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % accum_iter_determined_from_batch_size)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )

    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    optimizer = torch.optim._multi_tensor.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    # Loads model from checkpoint if specified
    # even though it doesn't assign anything to model, it does assign to model_without_ddp
    # which changes model under the hood
    # use: --resume="path/to/checkpoint.pth"
    print("loading model")
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # NOTE can comment this to skip training

        # TODO uncomment this block between the two notes to train
        # if not (test_image or test_video):
        #     train_stats = train_one_epoch(
        #         model,
        #         data_loader_train,
        #         args.accum_iter,
        #         optimizer,
        #         device,
        #         epoch,
        #         loss_scaler,
        #         log_writer=log_writer,
        #         args=args,
        #         fp32=args.fp32,
        #     )

        #     if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
        #         checkpoint_path = misc.save_model(
        #             args=args,
        #             model=model,
        #             model_without_ddp=model_without_ddp,
        #             optimizer=optimizer,
        #             loss_scaler=loss_scaler,
        #             epoch=epoch,
        #         )

        #     log_stats = {
        #         **{f"train_{k}": v for k, v in train_stats.items()},
        #         "epoch": epoch,
        #     }

        #     if args.output_dir and misc.is_main_process():
        #         if log_writer is not None:
        #             log_writer.flush()
        #         with pathmgr.open(
        #                 f"{args.output_dir}/log.txt",
        #                 "a",
        #         ) as f:
        #             f.write(json.dumps(log_stats) + "\n")

        if misc.is_main_process():
            # wandb.log(log_stats) TODO uncomment before training or data won't be saved
            visualize_prompting(model, args.video_prompts_dir, args.image_prompts_dir)

        print("Done loop on epoch {}".format(epoch))
        exit()
        ### End evaluation

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
        local_rank,
        shard_rank,
        num_gpus_per_node,
        num_shards,
        init_method,
        output_path,
        opts,
        stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)