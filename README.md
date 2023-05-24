# Video Prompting Readme

## Availiable Machines in Darrell Cluster
fangtooth - gpus 0,2,3,4,5,6,7,8
kraken - all gpus (there are 2 big 48GB gpus) - if running on these, please remember to use accumulate_iter.

## Install env 
``conda env create -f environment.yml``
This should take care of most dependencies.
``pip install pytorch-lightning==1.6.2 einops==0.6.1 omegaconf==2.3.0``
Installing additional dependencies for VQGAN. 

## Setup for VQGAN
Download pretrained VQGAN codebook checkpoint and config [vqgan_imagenet_f16_1024](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/?p=%2F), place both last.ckpt and model.yaml on the repository root (should be .../video_mae_code/)

## Setup up a datasets folder and create soft links to the datasets
```
mkdir -p ~/Datasets
ln -s /shared/group/ilsvrc ${HOME}/Datasets/ilsvrc
ln -s /shared/group/kinetics ${HOME}/Datasets/kinetics
ln -s /shared/amir/dataset/arxiv_resized_train_val_split ${HOME}/Datasets/arxiv_resized_train_val_split
```

Note that generally `/shared/group` network read time is faster than any other `/shared/` location, which very is slow. Therefore, if that dataset is not on `/shared/group` it is better to have a copy of it on the local storage, or ask amir to create a `/shared/group` path. For example, in fangtooth we have cvf stored on local storage:

`ln -s /home/amir/Datasets/arxiv_resized_train_val_split ${HOME}/Datasets/arxiv_resized_train_val_split`

## Example Pretraining command on a single node (8 gpus) on fangtooth:
export CUDA_VISIBLE_DEVICES="0,2,3,4,5,6,7,8" && output_dir=<my_output_dir> && python -m torch.distributed.launch --nproc_per_node=8 --use_env run_pretrain.py --log_dir ${output_dir}  --output_dir ${output_dir} --blr 1e-4 --dataset_root ${HOME}/Datasets
