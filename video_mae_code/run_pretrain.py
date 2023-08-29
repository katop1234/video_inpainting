# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from pathlib import Path
from main_pretrain import get_args_parser, main
import os
import torch.distributed as dist
import torch.multiprocessing as mp

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print('args: ', args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        print('Running Main')
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')
        try: 
            dist.destroy_process_group()
        except KeyboardInterrupt:
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
