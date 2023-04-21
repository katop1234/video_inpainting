# video_inpainting

code is in the video_mae_code
todo / ideas are in TODO.txt

From terminal run:
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 && torchrun --nproc_per_node=8 video_mae_code/run_pretrain.py
