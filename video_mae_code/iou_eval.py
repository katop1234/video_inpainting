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
    255,255,255,    # Index 1: White
]
single_object_cases = ['blackswan', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'libby', 'parkour']

def extract_image_seg(image_array, orig_height, orig_width, mask_type):
    #Extracts the segmentation (bottom half) of image_array and resizes to original size
    if mask_type == 'spatiotemporal':
        seg_array = image_array[112:, :, :]
    elif mask_type == '2x2 tube' or 'test image':
        seg_array = image_array[113:, 113:, :]
    
    seg_image = PIL.Image.fromarray(seg_array)
    seg_image = seg_image.resize((orig_height, orig_width))
    seg_array = np.asarray(seg_image)
    return seg_array

def extract_prompt_seg(video_array, orig_height, orig_width, mask_type):
    #Extracts the segmentation (bottom half) of the last 8 frames
    if mask_type == 'spatiotemporal':
        generated_segs = video_array[8:, :, :, :]
    elif mask_type == '2x2 tube':
        generated_segs = video_array
    elif mask_type == 'test image':
        generated_segs = video_array[:1, :, :, :]
        
    frames = []
    length = generated_segs.shape[0]
    for i in range(length):
        curr_image_array = generated_segs[i]
        curr_frame = extract_image_seg(curr_image_array, orig_height, orig_width, mask_type)
        frames.append(curr_frame)
    return frames
    
def create_segmentation(frame_array):
    #When the magnitude of the color at a certain pixel is above 80, the pixel will be red and otherwise black
    color_magnitude = np.linalg.norm(frame_array, axis=2)
    mask = color_magnitude > 80
    mask = mask.astype(np.uint8)
    return mask

def save_segmentations(frames, val, path, end_idx, mask_type):
    #Saves the segmentations at the proper frames
    num_frames = len(frames)
        
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

def load_model(model_path):
    model = MaskedAutoencoderViT()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model = model.to('cuda')
    return model  

def get_results_path(store_path, eval_name):
    results_path = os.path.join(store_path, eval_name)
    return results_path  

def generate_segmentations(model, store_path, single_prompt_csv, prompt_csv, davis_prompt_path, davis_2x2_prompt_path, davis_image_prompt_path):
    if type(model) is torch.nn.parallel.DistributedDataParallel:
        model = model.module
    
    results_orig_path = get_results_path(store_path, 'orig')
    results_2x2_path = get_results_path(store_path, '2x2')
    results_image_path = get_results_path(store_path, 'image')
    if not os.path.exists(results_orig_path):
        os.mkdir(results_orig_path)
    if not os.path.exists(results_2x2_path):
        os.mkdir(results_2x2_path)
    if not os.path.exists(results_image_path):
        os.mkdir(results_image_path)
    
    with open(prompt_csv, 'r') as file:
        csvreader = csv.reader(file)
        prompt_num = 0
        for row in csvreader:
            if row[0] != 'train':
                val = row[1]
                val_height = int(row[4])
                val_width = int(row[5])
                val_end_idx = int(row[7])
                
                video_prompt_orig = "DAVIS_{prompt_num}.mp4".format(prompt_num=prompt_num)

                video_prompt_orig = os.path.join(davis_prompt_path, video_prompt_orig)
                
                video_im_paste_orig = image_video_generation(video_prompt_orig, model, 'spatiotemporal')

                frames_orig = extract_prompt_seg(video_im_paste_orig[0], val_height, val_width, 'spatiotemporal')
 
                seg_save_orig_path = os.path.join(results_orig_path, val)

                if not os.path.exists(seg_save_orig_path):
                    os.mkdir(seg_save_orig_path)

                save_segmentations(frames_orig, val, seg_save_orig_path, val_end_idx, 'spatiotemporal')
                prompt_num += 1
    
    with open(single_prompt_csv, 'r') as file:
        csvreader = csv.reader(file)
        prompt_num = 0
        for row in csvreader:
            if row[0] != 'train':
                val = row[1]
                val_height = int(row[4])
                val_width = int(row[5])
                val_end_idx = int(row[7])
                
                video_prompt_2x2 = "DAVIS_2x2_{prompt_num}.mp4".format(prompt_num=prompt_num)
                image_prompt_2x2 = "DAVIS_image_{prompt_num}.png".format(prompt_num=prompt_num)
                
                video_prompt_2x2 = os.path.join(davis_2x2_prompt_path, video_prompt_2x2)
                image_prompt_2x2 = os.path.join(davis_image_prompt_path, image_prompt_2x2)
                
                video_im_paste_2x2 = image_video_generation(video_prompt_2x2, model, '2x2 tube')
                image_im_paste = image_video_generation(image_prompt_2x2, model, 'test image')
                
                frames_2x2 = extract_prompt_seg(video_im_paste_2x2[0], val_height, val_width, '2x2 tube')
                frames_image = extract_prompt_seg(image_im_paste[0], val_height, val_width, 'test image')
                
                seg_save_2x2_path = os.path.join(results_2x2_path, val)
                seg_save_image_path = os.path.join(results_image_path, val)
                
                if not os.path.exists(seg_save_2x2_path):
                    os.mkdir(seg_save_2x2_path)
                if not os.path.exists(seg_save_image_path):
                    os.mkdir(seg_save_image_path)
                    
                save_segmentations(frames_2x2, val, seg_save_2x2_path, val_end_idx, '2x2 tube')
                save_segmentations(frames_image, val, seg_save_image_path, val_end_idx, 'test image')
                
def image_video_generation(prompt_path, model, mask_test_type):
    test_model_input = get_test_model_input(file=prompt_path)
    test_model_input = spatial_sample_test_video(test_model_input)
    
    if mask_test_type == 'test image':
        _, test_model_output, mask = model(test_model_input, test_image=True)
    else: 
        _, test_model_output, mask = model(test_model_input, video_test_type=mask_test_type)

    num_patches = 14
    N = test_model_input.shape[0]

    test_model_output = test_model_output.view(N, -1, 196, 2, 1024)
    test_model_output = test_model_output.permute(0, 1, 3, 2, 4)
    test_model_output = test_model_output.flatten(1, 2)
    test_model_output = test_model_output.flatten(1, 2)

    y = test_model_output.argmax(dim=-1)
    im_paste, _, _ = decode_raw_prediction(mask, model, num_patches, test_model_input, y)
    im_paste = (im_paste.cpu().numpy()).astype(np.uint8)
    return im_paste
                
def run_evaluation_method(store_path):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    DAVIS_datasets_dir = os.path.join(parent_dir, 'DAVIS_datasets')
    
    davis_path_orig = os.path.join(DAVIS_datasets_dir, 'DAVIS_video_1')
    print('davis_path_orig: ', davis_path_orig)
    davis_path_2x2 = os.path.join(DAVIS_datasets_dir, 'DAVIS_video_2x2_single')
    print('davis_path_2x2: ', davis_path_2x2)
    davis_path_image = os.path.join(DAVIS_datasets_dir, 'DAVIS_image_single')
    print('davis_path_image: ', davis_path_image)
    
    # # davis_path_orig = '/shared/dannyt123/Datasets/DAVIS_video_1'
    # davis_path_orig = '/shared/dannyt123/video_inpainting/DAVIS_datasets/DAVIS_video_1'
    # # davis_path_2x2 = '/shared/dannyt123/Datasets/DAVIS_video_2x2_single'
    # davis_path_2x2 = '/shared/dannyt123/video_inpainting/DAVIS_datasets/DAVIS_video_2x2_single'
    # # davis_path_image = '/shared/dannyt123/Datasets/DAVIS_image_single'
    # davis_path_image = '/shared/dannyt123/video_inpainting/DAVIS_datasets/DAVIS_image_single'
    
    results_orig_path = get_results_path(store_path, 'orig')
    results_2x2_path = get_results_path(store_path, '2x2')
    results_image_path = get_results_path(store_path, 'image')
    
    single_mean_orig = evaluate.evaluation(results_orig_path, davis_path_orig, num_frames=8)
    single_mean_2x2 = evaluate.evaluation(results_2x2_path, davis_path_2x2, num_frames=16)
    single_mean_image = evaluate.evaluation(results_image_path, davis_path_image, num_frames=1)
    
    return single_mean_orig, single_mean_2x2, single_mean_image