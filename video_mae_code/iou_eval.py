import argparse
import csv
from models_mae import *
import numpy as np
import PIL
from PIL import Image
import os
import subprocess
import sys
import torch
from util.eval import *

sys.path.append('/shared/dannyt123/davis2017-evaluation')
import evaluate

#Constants
color_palette = [
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
        if i in indices and val in single_object_cases: #Temporarily for single object cases
            seg = frames[j]
            seg = create_segmentation(seg) #For temporarily while the model is still not very good
            j += 1
            seg_image = PIL.Image.fromarray(seg)
            seg_image = seg_image.convert('P')
            seg_image.putpalette(color_palette)
            curr_path = os.path.join(path, f'{i:05}.png')
            seg_image.save(curr_path)
        
def single_object_mean(per_sequence_csv):
    sum_mean = 0
    with open(per_sequence_csv, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            sequence_name = row[0][:-2]
            if sequence_name in single_object_cases:
                J_Mean = float(row[1])
                F_Mean = float(row[2])
                mean = (J_Mean + F_Mean) / 2.0
                sum_mean += mean
    return sum_mean / len(single_object_cases)
            
def global_mean(global_csv):
    with open(global_csv, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if row[0] != 'J&F-Mean':
                mean = row[0]
    return float(mean)

def load_model(model_path):
    model = MaskedAutoencoderViT()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model = model.to('cuda')
    return model  

def get_results_path(store_path, eval_name):
    results_path = os.path.join(store_path, eval_name)
    return results_path  

def generate_segmentations(model, store_path, eval_name, prompt_csv, davis_prompts_path):
    if type(model) is torch.nn.parallel.DistributedDataParallel:
        model = model.module
    
    results_path = get_results_path(store_path, eval_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    with open(prompt_csv, 'r') as file:
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
                prompt_path = os.path.join(davis_prompts_path, curr_prompt)
                
                test_model_input = get_test_model_input(file=prompt_path)
                test_model_input = spatial_sample_test_video(test_model_input)
                _, test_model_output, mask = model(test_model_input, test_spatiotemporal=True)
                
                num_patches = 14
                N = test_model_input.shape[0]
                
                test_model_input = test_model_input.view(N, -1, 196, 2, 1024)
                test_model_input = test_model_input.permute(0, 1, 3, 2, 4)
                test_model_input = test_model_input.flatten(1, 2)
                test_model_input = test_model_input.flatten(1, 2)
                
                y = test_model_output.argmax(dim=-1)
                im_paste, _, _ = decode_raw_prediction(mask, model, num_patches, test_model_input, y)

                im_paste = (im_paste.cpu().numpy()).astype(np.uint8)
                frames = extract_prompt_seg(im_paste[0], val_height, val_width)
                seg_save_path = os.path.join(results_path, val)
                
                if not os.path.exists(seg_save_path):
                    os.mkdir(seg_save_path)

                save_segmentations(frames, val, seg_save_path, val_end_idx, val_height, val_width)
                
def run_evaluation_method(store_path, eval_name, davis_path):
    results_path = get_results_path(store_path, eval_name)
    single_mean = evaluate.evaluation(results_path, davis_path)
    return single_mean