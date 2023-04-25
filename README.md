
Pretraining command:
fangtooth 1st run:
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 && torchrun --nproc_per_node=8 video_mae_code/run_pretrain.py
fangtooth 2nd run (diff port):
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 && torchrun --nproc_per_node=8 --master_port=29501 video_mae_code/run_pretrain.py

fangtooth resume
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 && torchrun --nproc_per_node=8 video_mae_code/run_pretrain.py --resume="/shared/katop1234/video_inpainting/video_inpainting/video_mae_code/output_dir/checkpoint-00085.pth"

jormunngandr 1st run:
  torchrun --nproc_per_node=8 video_mae_code/run_pretrain.py --batch_size=1

jormunngandr 2nd run:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && torchrun --nproc_per_node=8 --master_port=29501 video_mae_code/run_pretrain.py --batch_size=1

Finishing touches:
**try Rotary Positional Embedding (like in LLAMA)**
consider using rotary positional embedding for the video

Can also try the alibi stuff
