# Video Prompting Readme

## Availiable Machines in Darrell Cluster
fangtooth - gpus 0,2,3,4,5,6,7,8
kraken - all gpus (there are 2 big 48GB gpus) - if running on these, please remember to use accumulate_iter.

## Install env 
``conda env create -f environment.yml``
This should take care of most dependencies.

## Example Pretraining command on a single node (8 gpus) on fangtooth:
export CUDA_VISIBLE_DEVICES="0,2,3,4,5,6,7,8" && output_dir=<my_output_dir> && python -m torch.distributed.launch --nproc_per_node=8 --use_env run_pretrain.py --log_dir ${output_dir}  --output_dir ${output_dir} --blr 1e-4
