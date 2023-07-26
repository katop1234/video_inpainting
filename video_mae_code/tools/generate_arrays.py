import argparse
import os
import sys
import torch
from pathlib import Path
sys.path.append('/shared/dannyt123/video_inpainting/video_mae_code/')
from models_mae import *
from util.eval import *


PARSER = argparse.ArgumentParser()
#
PARSER.add_argument('--checkpoint_path', type=str, default='/shared/dannyt123/video_inpainting/output_dir/two_patch_kinetics_3/checkpoint-00725.pth', 
                    help='directory of the checkpoint')
#
PARSER.add_argument('--videos_dir', type=str, default='/shared/dannyt123/video_inpainting/test_videos/bair_evaluation/test_videos', 
                    help='directory of the checkpoint')

def load_model(model_path):
    model = MaskedAutoencoderViT()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model = model.to('cuda')
    return model  

def generate_arrays(checkpoint_path, videos_dir):
  model = load_model(checkpoint_path)
  checkpoint_store_path = Path(checkpoint_path).parent.absolute()
  generated_arrays_path = os.path.join(checkpoint_store_path, "fvd_arrays")
  zero_path = os.path.join(generated_arrays_path, '0')
  if not os.path.exists(generated_arrays_path):
    os.mkdir(generated_arrays_path)
  
  if not os.path.exists(zero_path):
    os.mkdir(zero_path)
  
  for video in os.listdir(videos_dir):
    video_file = os.path.join(videos_dir, video)
    np_name = "{video_name}.npy".format(video_name=video[:-4])
    np_path = os.path.join(zero_path, np_name)
    
    test_model_input = get_test_model_input(file=video_file)
    test_model_input = spatial_sample_test_video(test_model_input)
    _, test_model_output, mask = model(test_model_input, video_test_type="frame interpolation")
    im_paste, _ = video_generation(model, mask, test_model_input, test_model_output)
    np_array = im_paste.transpose(0, 1, 3, 4, 2).squeeze()
    
    save_dict = {'video': np_array}
    np.save(np_path, save_dict, allow_pickle=True)
  
def main():
  args = PARSER.parse_known_args()[0]
  
  generate_arrays(args.checkpoint_path, args.videos_dir)

if __name__ == '__main__':
  main()