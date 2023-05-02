import torch
import wandb
import os
import numpy as np
from kinetics_and_images import (get_test_model_input_from_kinetics, 
                                 save_test_output, save_frames_as_mp4)
import models_mae
from engine_pretrain import normalized_to_uint8, get_test_model_input, spatial_sample_test_video

def reconstruct(mask, ground_truth, test_model_output):
    assert mask.shape == ground_truth.shape[:2] == test_model_output.shape[:2], "Input dimensions must match"
    
    # Expand the mask tensor to match the dimensions of the other tensors
    expanded_mask = mask.unsqueeze(-1).expand_as(ground_truth)
    
    # Use the mask to select values from test_model_output (mask=1) or ground_truth (mask=0)
    result = torch.where(expanded_mask == 1, test_model_output, ground_truth)
    
    return result

@torch.no_grad()
def visualize_prompting(model, input_video_viz_dir, input_image_viz_dir):
    model.eval()
    visualize_image_prompting(model, input_image_viz_dir)
    visualize_video_prompting(model, input_video_viz_dir)
    model.train()

@torch.no_grad()
def visualize_image_prompting(model, input_image_viz_dir):
    
    if not os.path.exists(input_image_viz_dir):
        input_image_viz_dir = "/shared/katop1234/video_inpainting/video_inpainting/test_cases/visual_prompting_images/"
        print("Using default input_image_viz_dir: ", input_image_viz_dir)
        
    ### Test on images
    for i, img_file in enumerate(os.listdir(input_image_viz_dir)):
        img_file = os.path.join(input_image_viz_dir, img_file) 
        test_model_input = get_test_model_input(file=img_file)
        test_model_input = test_model_input.cuda()

        with torch.no_grad():
            # TODO why does it mask it weirdly, like only the bottom 1/4 even after i fixed the mask_test_image function??
            _, test_model_output, mask = model(test_model_input, test_image=True)
        
        if type(model) is torch.nn.parallel.DistributedDataParallel:
            patchified_gt = model.module.patchify(test_model_input)
            reconstructed_output = reconstruct(mask, patchified_gt, test_model_output)
            test_model_output = model.module.unpatchify(reconstructed_output)
        elif type(model) is models_mae.MaskedAutoencoderViT:
            patchified_gt = model.patchify(test_model_input)
            reconstructed_output = reconstruct(mask, patchified_gt, test_model_output)
            test_model_output = model.unpatchify(reconstructed_output)
        else:
            raise NotImplementedError("Something's funky")

        # TODO USE SINGLE MEAN STD FOR IMG + VIDEO, see wherever normalized_to_uint8 and inverse is called
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        test_model_output = normalized_to_uint8(test_model_output, mean, std)

        # Rearrange dimensions to have channels last, and remove unnecessary dimensions
        denormalized_img = test_model_output.squeeze(0).permute(1, 2, 3, 0).squeeze(0) # (224, 224, 3)

        # Convert to numpy array, scale back to [0, 255] and convert to uint8 data type
        image_array = (denormalized_img.cpu().numpy()).astype(np.uint8)

        output_img_name = 'test_model_output_img' + str(i) + '.png'

        save_test_output(image_array, output_img_name)

        image = wandb.Image(image_array)

        wandb.log({output_img_name: image})

@torch.no_grad()
def visualize_video_prompting(model, input_video_viz_dir="test_cases/final_temporal_videos/"):
    test_model_input = get_test_model_input(data_dir=input_video_viz_dir) # DEBUG check range and shape at each step

    save_frames_as_mp4(normalized_to_uint8(test_model_input), file_name="test_input_video.mp4")

    test_model_input = spatial_sample_test_video(test_model_input)

    with torch.no_grad():
        # TODO change test_temporal to True later
        _, test_model_output, _ = model(test_model_input)

    if type(model) is torch.nn.parallel.DistributedDataParallel:
        test_model_output = model.module.unpatchify(test_model_output)
    elif type(model) is models_mae.MaskedAutoencoderViT:
        test_model_output = model.unpatchify(test_model_output)
    else:
        raise NotImplementedError("Something's funky")

    test_model_output = normalized_to_uint8(test_model_output)

    save_test_output(test_model_output, name="test_output_video.mp4")

    wandb_video_object = wandb.Video(
            data_or_path="test_output_video.mp4",
            fps=30,
            )
    wandb.log({"video": wandb_video_object})
