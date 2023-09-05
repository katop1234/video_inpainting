
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
# import timm  # do not uncomment
import torch
import torch.backends.cudnn as cudnn
import traceback
from iopath.common.file_io import g_pathmgr as pathmgr
import models_mae
from mae_image import models_mae_image
from engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.tensorboard import SummaryWriter
import util.decoder.utils as utils
from iou_eval import generate_segmentations, run_evaluation_method, generate_colorizations
import evaluate_colorization
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)

    parser.add_argument("--test_mode", action="store_true", help="If provided, skips training then exits")
    parser.add_argument("--batch_size_image", default=64, type=int, help="Image batch size per GPU")
    parser.add_argument("--batch_size_video", default=4, type=int, help="Video batch size per GPU")
    parser.add_argument("--epochs", default=4000, type=int)
    parser.add_argument("--accum_iter_image", default=1, type=int, help="accum iteration for image")
    parser.add_argument("--accum_iter_video", default=64, type=int, help="accum iteration for video")
    
    #Wandb
    parser.add_argument(
        "--wandb_resume",
        action='store_true',
        help="Provide for resuming a wandb run",
    )
    
    parser.add_argument(
        "--wandb_id",
        default="",
        type=str,
        help="Wandb ID for resuming a run",
    )
    
    #Training
    parser.add_argument(
        "--no_cont_pretrain",
        action='store_true',
        help="Provide for restarting the optimizer and epoch count",
    )
    
    parser.add_argument(
        "--train_cct_only",
        action='store_true',
        help="Provide for training cct encoder/decoder blocks only",
    )
    
    parser.add_argument(
        "--train_video_only",
        action='store_true',
        help="Provide for training video encoder/decoder blocks only",
    )


    # Model parameters
    parser.add_argument(
        "--model",
        # default="mae_vit_large_patch16",
        default ='mae_blank',
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    
    parser.add_argument(
        "--video_encoder_indices",
        default="",
        type=str,
        help="Indice placement of video encoder blocks, e.g 1,2,3. Number of indices must be the same as video_encoder_depth",
    )
    
    parser.add_argument(
        "--video_decoder_indices",
        default="",
        type=str,
        help="Indice placement of video decoder blocks, e.g 1,2,3. Number of indices must be the same as video_decoder_depth",
    )
    
    parser.add_argument(
        "--s_transfer_encoder_indices",
        default="",
        type=str,
        help="Indice placement of spatial encoder blocks transfer for X-CLIP or AIM, e.g 1,2,3.",
    )
    
    parser.add_argument(
        "--s_transfer_decoder_indices",
        default="",
        type=str,
        help="Indice placement of spatial decoder blocks transfer for X-CLIP or AIM, e.g 1,2,3.",
    )
    
    parser.add_argument(
        "--random_video",
        action='store_true',
        help="Provide for randomnly moving the video blocks",
    )
    
    parser.add_argument(
        "--X_CLIP",
        action='store_true',
        help="Provide for using X_CLIP",
    )
    
    parser.add_argument(
        "--AIM",
        action='store_true',
        help="Provide for using AIM",
    )
    
    parser.add_argument(
        "--transfer_encoder_depth",
        default=0,
        type=int,
        help="The number of CCT blocks in the encoder from the last encoders",
    )
    
    parser.add_argument(
        "--transfer_decoder_depth",
        default=0,
        type=int,
        help="The number of CCT blocks in the decoder from the last decoders",
    )
    
    parser.add_argument(
        "--video_encoder_depth",
        default=0,
        type=int,
        help="The number of video blocks in the encoder from the last encoders",
    )
    
    parser.add_argument(
        "--video_decoder_depth",
        default=0,
        type=int,
        help="The number of video blocks in the decoder from the last decoders",
    )

    parser.add_argument(
        "--mae_image",
        action='store_true',
        help="Provide for mae_image",
    )
    
    parser.add_argument(
        "--decoder_masking",
        action='store_true',
        help="Provide for decoder_masking",
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

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    
    parser.add_argument(
        "--use_checkpointing",
        action='store_true',
        help="Use Checkpoint blocks",
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
        "--warmup_epochs", type=int, default=120, metavar="N", help="epochs to warmup LR" #5
    )
    
    parser.add_argument(
        "--new_faster_lr",
        action='store_true',
        help="Provide for faster learning rate for new parameters in X_CLIP",
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

    parser.add_argument("--embed_dim", default=1024, type=int)
    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=4, type=int)
    parser.add_argument("--depth", default=24, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=5, type=int)
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
        # type=float,
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
    parser.add_argument('--image_dataset_list', nargs='+', default=['imagenet,cvf'])
    parser.add_argument('--image_dataset_conf', nargs='+', default=['1,1']) 
    parser.add_argument('--video_dataset_list', nargs='+', default=["kinetics,Objectron,SSV2,UCF101,CSV"])
    parser.add_argument('--video_dataset_conf', nargs='+', default=['2,2,1,1,2'])
    parser.add_argument('--image_video_ratio', default=0.0, help='default means equally mixed between the two')

    parser.add_argument('--davis_eval_freq', default=1, help='frequency of computing davis eval metrics')
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
    if type(args.image_itr) == str:
        args.image_itr = int(args.image_itr)
        print('args.image_itr: ', image_itr)
    if type(args.video_itr) == str:
        args.video_itr = int(args.video_itr)
        print('args.video_itr: ', args.video_itr)
    if type(args.davis_eval_freq) == str:
        args.davis_eval_freq = int(args.davis_eval_freq)
        
    if args.image_itr > 0:
        dataset_image_train = MergedDataset(args.dataset_root, args.image_dataset_list, args.image_dataset_conf, 'image')
    else:
        dataset_image_train = None
    
    if args.video_itr > 0:
        print("creating video merged dataset")
        print("args.video_dataset_list: ", args.video_dataset_list)
        dataset_video_train = MergedDataset(args.dataset_root, args.video_dataset_list, args.video_dataset_conf, 'video')
    else:
        dataset_video_train = None
    
    num_tasks = misc.get_world_size()  # 8 gpus
    global_rank = misc.get_rank()
    
    if args.image_itr > 0:
        sampler_image_train = torch.utils.data.DistributedSampler(
            dataset_image_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_image_train = None

    if args.video_itr > 0:
        sampler_video_train = torch.utils.data.DistributedSampler(
            dataset_video_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else: 
        sampler_video_train = None

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

    if args.image_itr > 0:
        data_loader_image_train = torch.utils.data.DataLoader(
            dataset_image_train,
            sampler=sampler_image_train,
            batch_size=args.batch_size_image,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_image_train = None

    # args.batch_size_video = 4
    print('args.batch_size_video: ', 4)
    if args.video_itr > 0: 
        data_loader_video_train = torch.utils.data.DataLoader(
            dataset_video_train,
            sampler=sampler_video_train,
            batch_size=args.batch_size_video,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_video_train = None


    # define the model
    if args.mae_image:
        model = models_mae_image.__dict__[args.model](
            **vars(args),
        )
    else:
        print('not in mae_image')
        model = models_mae.__dict__[args.model](
            **vars(args),
        )

    try:
        model.to(device)
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()

        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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
            # find_unused_parameters=True,
            find_unused_parameters=False,
            static_graph=True,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    if args.X_CLIP:
        param_groups = misc.add_weight_decay_and_lr(
            model_without_ddp,
            args.lr,
            args.weight_decay,
            bias_wd=args.bias_wd,
        )
    else:
        param_groups = misc.add_weight_decay(
            model_without_ddp,
            args.weight_decay,
            bias_wd=args.bias_wd,
        )

    if args.beta is None:
        beta = (0.9, 0.95)
    else: 
        beta = tuple(float(x) for x in args.beta[0].split(','))
        
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
    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if misc.is_main_process():
        wandb_config = vars(args)
        base_lr = (args.lr * 256 / eff_batch_size)
        wandb_config['base_lr'] = base_lr
        if args.wandb_resume:
            print("in wandb_resume")
            wandb.init(
            project="video_inpainting2",
            config=wandb_config,
            resume=True,
            id=args.wandb_id,
            )
        else:
            wandb.init(
                project="video_inpainting2",
                config=wandb_config,
                )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    combined_dataloader = CombinedGen(data_loader_image_train, data_loader_video_train, args.accum_iter_image, args.accum_iter_video, args.image_itr, args.video_itr)

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            if data_loader_image_train:
                data_loader_image_train.sampler.set_epoch(epoch)
            if data_loader_video_train:
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

        if epoch % args.davis_eval_freq == 0 and misc.is_main_process():
            with torch.no_grad():
                if args.test_mode:
                    log_stats = {}
                    
                model.eval()
                ## Segmentation specific code
                store_path = os.path.join(args.output_dir, "davis_segs")
                if not os.path.exists(store_path):
                    os.mkdir(store_path)
                
                parent = Path(__file__).parent.absolute()
                prompt_csv = os.path.join(parent, "datasets/davis_prompt.csv")
                single_prompt_csv = os.path.join(parent, "datasets/davis_single_prompt.csv")
                
                davis_prompt_path = os.path.join(parent, "../test_videos/davis_prompt")
                davis_2x2_prompt_path = os.path.join(parent, "../test_videos/davis_2x2_single_prompt")
                davis_image_prompt_path = os.path.join(parent, "../test_images/single_davis_image_prompts")
                
                generate_segmentations(model, store_path, single_prompt_csv, prompt_csv, davis_prompt_path, davis_2x2_prompt_path, davis_image_prompt_path, mae_image=args.mae_image)
                print("Finished Saving Davis Eval Segmentations")
                
                # single_mean_orig, single_mean_2x2, single_mean_image = run_evaluation_method(store_path)
                single_mean_2x2, single_mean_image = run_evaluation_method(store_path)
                # log_stats["Davis_single_mean_orig"] = single_mean_orig
                print('single_mean_2x2: ', single_mean_2x2)
                print('single_mean_image: ', single_mean_image)
                log_stats["single_mean_2x2"] = single_mean_2x2
                log_stats["single_mean_image"] = single_mean_image
                
                ## Colorization specific code
                store_path = os.path.join(args.output_dir, "davis_cols")
                if not os.path.exists(store_path):
                    os.mkdir(store_path)
                
                davis_prompt_path = os.path.join(parent, "../test_videos/colorization_davis_prompt") # TODO not supported yet
                print("Colorization eval for /test_videos/colorization_davis_prompt not supported yet")
                davis_2x2_prompt_path = os.path.join(parent, "../test_videos/colorization_davis_2x2_single_prompt")
                davis_image_prompt_path = os.path.join(parent, "../test_images/colorization_single_davis_image_prompts")
                  
                generate_colorizations(model, store_path, single_prompt_csv, prompt_csv, davis_prompt_path, davis_2x2_prompt_path, davis_image_prompt_path, mae_image=args.mae_image)
                print("Finished Saving Colorization examples")       
                
                single_mean_2x2, single_mean_image = evaluate_colorization.run_evaluation_method(store_path)
                print('single_mean_2x2:', single_mean_2x2)
                print('single_mean_image:', single_mean_image)

                log_stats = {}  # Assuming log_stats was previously defined
                log_stats["single_mean_2x2"] = single_mean_2x2
                log_stats["single_mean_image"] = single_mean_image
                
                model.train()

        if misc.is_main_process():
            if not args.test_mode:
                wandb.log(log_stats)
            model.eval()
            parent = Path(__file__).parent.absolute()
            video_prompts_dir = os.path.join(parent, "../test_cases")
            visualize_prompting(model, video_prompts_dir, mae_image=args.mae_image, mask_ratio_image=args.mask_ratio_image, mask_ratio_video=args.mask_ratio_video)
            model.train()
        print("Done loop on epoch {}".format(epoch))

        if args.test_mode:
            exit()
        ### End evaluation

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]
