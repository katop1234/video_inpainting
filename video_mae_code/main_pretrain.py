
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
import json
import datetime
import wandb
import os
import time
from dataset_factory import MergedDataset, CombinedGen
from util.eval import visualize_prompting
import util.env  # do not uncomment
import util.misc as misc
import numpy as np
import timm  # do not uncomment
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
import models_mae
from engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument("--test_mode", action="store_true", help="If provided, skips training then exits")
    parser.add_argument("--batch_size_image", default=64, type=int, help="Image batch size per GPU")
    parser.add_argument("--batch_size_video", default=1, type=int, help="Video batch size per GPU")
    parser.add_argument("--epochs", default=4000, type=int)
    parser.add_argument("--accum_iter_image", default=1, type=int, help="accum iteration for image")
    parser.add_argument("--accum_iter_video", default=64, type=int, help="accum iteration for video")

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
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--video_prompts_dir",
        default="/shared/katop1234/video_inpainting/video_inpainting/test_cases/random_masked_videos/",
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
    parser.add_argument("--decoder_depth", default=4, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=1, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int) # save every epoch for safety
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=1, type=int, help="We set this to 2 by default in dataset_factory.get_dataset for Kinetics.")
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=0.02,
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

    parser.add_argument("--dataset_root", default=os.path.join(os.path.expanduser("~"), "Datasets"), help="parent folder for all datasets")
    parser.add_argument('--image_dataset_list', nargs='+', default=['cvf'])
    parser.add_argument('--image_dataset_conf', nargs='+', default=[1]) 
    parser.add_argument('--video_dataset_list', nargs='+', default=['kinetics'])
    parser.add_argument('--video_dataset_conf', nargs='+', default=[1])
    parser.add_argument('--image_itr', default=4, type=int, help='number of image only itr')
    parser.add_argument('--video_itr', default=1, type=int, help='number of video only itr')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Dataset combining image and video data
    dataset_image_train = MergedDataset(args.dataset_root, args.image_dataset_list, args.image_dataset_conf, 'image')
    dataset_video_train = MergedDataset(args.dataset_root, args.video_dataset_list, args.video_dataset_conf, 'video')

    num_tasks = misc.get_world_size()  # 8 gpus
    global_rank = misc.get_rank()
    
    sampler_image_train = torch.utils.data.DistributedSampler(
        dataset_image_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    sampler_video_train = torch.utils.data.DistributedSampler(
        dataset_video_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )


    print("Sampler_train = %s" % str(sampler_image_train))

    if global_rank == 0 and args.log_dir is not None:
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print("Batch size image is", args.batch_size_image)
    print("Batch size video is", args.batch_size_video)
    print("Num GPUs is", misc.get_world_size())

    data_loader_image_train = torch.utils.data.DataLoader(
        dataset_image_train,
        sampler=sampler_image_train,
        batch_size=args.batch_size_image,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_video_train = torch.utils.data.DataLoader(
        dataset_video_train,
        sampler=sampler_video_train,
        batch_size=args.batch_size_video,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    # define the model
    model = models_mae.__dict__[args.model](
        **vars(args),
    )

    try:
        model.to(device)
    except:
        print("bugged out moving model to gpu")
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        exit()

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # We compute effective batch size based on images
    eff_batch_size = args.batch_size_image * args.accum_iter_image * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations images: %d" % args.accum_iter_image)
    print("accumulate grad iterations videos: %d" % args.accum_iter_video)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=True, #True
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

    print("loading model")
    _ = misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if misc.is_main_process():
        wandb_config = vars(args)
        base_lr = (args.lr * 256 / eff_batch_size)
        wandb_config['base_lr'] = base_lr
        wandb.init(
            project="video_inpainting2",
            config=wandb_config)

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    combined_dataloader = CombinedGen(data_loader_image_train, data_loader_video_train, args.accum_iter_image, args.accum_iter_video, args.image_itr, args.video_itr)

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_image_train.sampler.set_epoch(epoch)
            data_loader_video_train.sampler.set_epoch(epoch)

        if not args.test_mode:
            train_stats = train_one_epoch(
                model,
                combined_dataloader,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
                fp32=args.fp32,
            )

            if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
                checkpoint_path = misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with pathmgr.open(
                        f"{args.output_dir}/log.txt",
                        "a",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        dir_path = os.path.dirname(os.path.realpath(__file__))
        image_prompts_dir = os.path.join(dir_path, "../test_images")
        if misc.is_main_process():
            if not args.test_mode:
                wandb.log(log_stats)
            visualize_prompting(model, image_prompts_dir, args.video_prompts_dir)

        print("Done loop on epoch {}".format(epoch))

        if args.test_mode:
            exit()
        ### End evaluation

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]
