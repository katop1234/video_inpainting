import argparse
import csv
from models_mae import *
import numpy as np
import PIL
from PIL import Image
import os
import torch
from util.eval import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='Path to model for evaluation.',
                    default='/shared/dannyt123/video_inpainting/output_dir/kinetics_cvf/checkpoint-00052.pth')
parser.add_argument('--davis_path', type=str, help='Path to the DAVIS folder containing the JPEGImages, Annotations, '
                                                   'ImageSets, Annotations_unsupervised folders',
                    default='/shared/dannyt123/Datasets/DAVIS_trainval')
parser.add_argument('--davis_eval_path', type=str, help='Path to the davis2017-evaluation folder',
                    default='/shared/dannyt123/davis2017-evaluation')
parser.add_argument('--davis_prompts_path', type=str, help='Path to the folder containing all the DAVIS video prompts',
                    default='/shared/dannyt123/video_inpainting/test_videos/Davis_Prompt')
parser.add_argument('--eval_name', type=str, help='Name for the evaluation computation',
                    default='model_mae')
parser.add_argument('--prompt_csv', type=str, help='Path to the csv file containing the information about the DAVIS prompts',
                    default='/shared/dannyt123/video_inpainting/video_mae_code/datasets/davis_prompt.csv')

#Constants
palette = [
    0,   0,   0,    # Index 0: Black
    128, 0,   0,    # Index 1: Red
]
single_object_cases = ['blackswan', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'libby', 'parkour']

def extract_image_seg(image_array, orig_height, orig_width):
    #Extracts the segmentation (bottom half) of image_array and resizes to original size
    seg_array = image_array[112:, :, :]
    seg_image = PIL.Image.fromarray(seg_array)
    seg_image = seg_image.resize((orig_height, orig_width))
    seg_array = np.asarray(seg_image)
    return seg_array

def extract_prompt_seg(video_array, orig_height, orig_width):
    #Extracts the segmentation (bottom half) of the last 8 frames
    generated_segs = video_array[8:, :, :, :]
    frames = []
    for i in range(8):
        curr_image_array = generated_segs[i]
        curr_frame = extract_image_seg(curr_image_array, orig_height, orig_width)
        frames.append(curr_frame)  
    return frames
        
def black_seg(orig_height, orig_width):
    #Returns black image of original size
    final = np.zeros((orig_width, orig_height), dtype=np.uint8)
    return final
    
def create_segmentation(frame_array):
    #When the magnitude of the color at a certain pixel is above 80, the pixel will be red and otherwise black
    color_magnitude = np.linalg.norm(frame_array, axis=2)
    mask = color_magnitude > 80
    mask = mask.astype(np.uint8)
    return mask

def save_segmentations(frames, val, path, end_idx, orig_height, orig_width):
    #Saves the segmentations at the proper frames and black segmentations else where
    num_frames = 8
    sampling_rate = end_idx // num_frames
    indices = []
    for i in range(num_frames):
        indices.append(i * sampling_rate)
    
    j = 0
    for i in range(end_idx + 1):
        if i in indices and val in single_object_cases:
            seg = frames[j]
            seg = create_segmentation(seg) #For temporarily while the model is still not very good
            j += 1
        else:
            seg = black_seg(orig_height, orig_width)
        
        seg_image = PIL.Image.fromarray(seg)
        seg_image = seg_image.convert('P')
        seg_image.putpalette(palette)
        curr_path = os.path.join(path, f'{i:05}.png')
        seg_image.save(curr_path)
    
def main():
    args = parser.parse_known_args()[0]
    
    #Creating results_path
    unsupervised_path = os.path.join(args.davis_eval_path, "results/unsupervised")
    results_path = os.path.join(unsupervised_path, args.eval_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    #Loading Model
    model = MaskedAutoencoderViT()
    model.load_state_dict(torch.load(args.model_path)['model'])
    model.eval()
    model = model.to('cuda')

    with open(args.prompt_csv, 'r') as file:
        csvreader = csv.reader(file)
        prompt_num = 0
        for row in csvreader:
            if row[0] != 'train':
                val = row[1]
                val_height = int(row[4])
                val_width = int(row[5])
                val_end_idx = int(row[7])
                
                curr_prompt = "DAVIS_{prompt_num}.mp4".format(prompt_num=prompt_num)
                prompt_num += 1
                prompt_path = os.path.join(args.davis_prompts_path, curr_prompt)
                
                test_model_input = get_test_model_input(file=prompt_path)
                test_model_input = spatial_sample_test_video(test_model_input)
                _, test_model_output, mask = model(test_model_input, test_spatiotemporal=True)
                
                num_patches = 14
                y = test_model_output.argmax(dim=-1)
                im_paste, _, _ = decode_raw_prediction(mask, model, num_patches, test_model_input, y)

                im_paste = (im_paste.cpu().numpy()).astype(np.uint8)
                frames = extract_prompt_seg(im_paste[0], val_height, val_width)
                seg_save_path = os.path.join(results_path, val)
                
                if not os.path.exists(seg_save_path):
                    os.mkdir(seg_save_path)

                save_segmentations(frames, val, seg_save_path, val_end_idx, val_height, val_width)
                print("Saved segmentations at {seg_save_path}".format(seg_save_path=seg_save_path))
                
    run_path = os.path.join(args.davis_eval_path, "evaluation_method.py")
    run_command = "python3 {run_path} --davis_path {davis_path} --task unsupervised --results_path {results_path}".format(run_path=run_path, davis_path=args.davis_path, results_path=results_path)
    os.system(run_command)
                
if __name__ == "__main__":
    main()