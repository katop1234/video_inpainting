# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import math
import os
import time
from collections import defaultdict, deque, OrderedDict

import util.logging as logging
import psutil
import torch
import torch.distributed as dist
# import torch.fb.rendezvous.zeus
from iopath.common.file_io import g_pathmgr as pathmgr
from util.logging import master_print as print
from torch import inf


logger = logging.get_logger(__name__)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, path):
    if is_main_process():
        print(f"save path {path}")
        with pathmgr.open(path, "wb") as f:
            torch.save(state, f)


def init_distributed_mode(args):
    if args.no_env:
        pass
    elif args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, fp32=False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):  
        
        # print('before scale backward')
        self._scaler.scale(loss).backward(create_graph=create_graph)

        # print('before update_grad check')
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # print('before unscale')
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                # print('after unscale')
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                # print('after norm')
            else:
                # print('before unscale in else')
                self._scaler.unscale_(optimizer)
                # print('after unscale in else')
                norm = get_grad_norm_(parameters)
                # print('after norm in else')
            # print('before step')
            self._scaler.step(optimizer)
            # print('after step')
            self._scaler.update()
            # print('after update')
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    checkpoint_path = "{}/checkpoint-{:05d}.pth".format(args.output_dir, epoch)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path


def get_last_checkpoint(args):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = args.output_dir
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        return os.path.join(d, name)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if not args.resume:
        args.resume = get_last_checkpoint(args)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            with pathmgr.open(args.resume, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        
        if (args.X_CLIP or args.AIM) and args.no_cont_pretrain:
            for j in range(args.transfer_encoder_depth):
                i = (args.depth - args.transfer_encoder_depth) + j
                model_without_ddp.blocks[i].attn.in_proj_weight = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.qkv.weight'.format(i=i)].to(device=args.device))
                model_without_ddp.blocks[i].attn.in_proj_bias = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.qkv.bias'.format(i=i)].to(device=args.device))
                model_without_ddp.blocks[i].attn.out_proj.weight = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.proj.weight'.format(i=i)].to(device=args.device))
                model_without_ddp.blocks[i].attn.out_proj.bias = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.proj.bias'.format(i=i)].to(device=args.device))
                print('set weights in cct encoder for: ', i)
                
            # for i in range(args.depth):
            #     if i % 2 == 1:
            #     # if True:
            #         model_without_ddp.blocks[i].attn.in_proj_weight = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.qkv.weight'.format(i=i)].to(device=args.device))
            #         model_without_ddp.blocks[i].attn.in_proj_bias = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.qkv.bias'.format(i=i)].to(device=args.device))
            #         model_without_ddp.blocks[i].attn.out_proj.weight = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.proj.weight'.format(i=i)].to(device=args.device))
            #         model_without_ddp.blocks[i].attn.out_proj.bias = torch.nn.Parameter(checkpoint['model']['blocks.{i}.attn.proj.bias'.format(i=i)].to(device=args.device))
            #         print('set weights in cct encoder for: ', i)
            
                
            for j in range(args.transfer_decoder_depth):
                i = (args.decoder_depth - args.transfer_decoder_depth) + j
                model_without_ddp.decoder_blocks[i].attn.in_proj_weight = torch.nn.Parameter(checkpoint['model']['decoder_blocks.{i}.attn.qkv.weight'.format(i=i)].to(device=args.device))
                model_without_ddp.decoder_blocks[i].attn.in_proj_bias = torch.nn.Parameter(checkpoint['model']['decoder_blocks.{i}.attn.qkv.bias'.format(i=i)].to(device=args.device))
                model_without_ddp.decoder_blocks[i].attn.out_proj.weight = torch.nn.Parameter(checkpoint['model']['decoder_blocks.{i}.attn.proj.weight'.format(i=i)].to(device=args.device))
                model_without_ddp.decoder_blocks[i].attn.out_proj.bias = torch.nn.Parameter(checkpoint['model']['decoder_blocks.{i}.attn.proj.bias'.format(i=i)].to(device=args.device))
                print('set weights in decoder for: ', i)

        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
            and not args.no_cont_pretrain
        ):
            print('not in no_cont_pretrain')
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")
        elif args.no_cont_pretrain:
            print("in no_cont_pretrain")
            args.start_epoch = 0
            
    return args.resume


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    

def add_weight_decay_and_lr(model, lr=8e-6, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    decay_faster = []
    no_decay = []
    no_decay_faster = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            if "message" in name:
                no_decay_faster.append(param)
            else:
                no_decay.append(param)
        else:
            if "message" in name:
                decay_faster.append(param)
            else:
                decay.append(param)
        
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": no_decay_faster, "weight_decay": 0.0, "lr": lr * 10},
        {"params": decay, "weight_decay": weight_decay},
        {"params": decay_faster, "weight_decay": weight_decay, "lr": lr * 10},
    ]


def inflate(model_2d, model_3d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "patch_embed.proj.weight" in k:
            v3d = model_3d[k]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            state_dict_inflated[k] = v3d.clone()
        elif "pos_embed" in k:
            pos_embed_cls, pos_embed_spatial = torch.split(v2d, [1, 196], dim=1)
            state_dict_inflated["pos_embed_cls"] = pos_embed_cls.clone()
            state_dict_inflated["pos_embed"] = pos_embed_spatial.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def convert_checkpoint(model_2d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "head.projection.weight" in k:
            state_dict_inflated["head.weight"] = v2d.clone()
        elif "head.projection.bias" in k:
            state_dict_inflated["head.bias"] = v2d.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated
