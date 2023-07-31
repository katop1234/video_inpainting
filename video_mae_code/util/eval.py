import torch
import wandb
import os
import numpy as np
import cv2
from PIL import Image
from util.decoder import utils
import torch.nn.functional as F
from torchvision import transforms
import util.decoder.constants as constants
import random

def save_frames_as_mp4(frames: torch.Tensor, file_name: str):
    '''
    "The input tensor should have the shape: (N, C, T, H, W)"
    '''
    frames_np = frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames_np = np.clip(frames_np, 0, 255)
    frames_uint8 = frames_np.astype(np.uint8)
    num_frames, height, width, _ = frames_uint8.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = cv2.cvtColor(frames_uint8[i], cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    return frames_uint8

def save_test_output(output, name):
    if output.shape == (1, 3, 16, 224, 224):
        return save_frames_as_mp4(output, name)
    elif output.shape == (224, 224, 3):
        return Image.fromarray(output).save(name)
    raise NotImplementedError

def get_random_file(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    random_file = random.choice(files)
    return os.path.join(data_dir, random_file)

def video_to_tensor(video_path, target_size=(224, 224), num_frames=16):
    resized_frames = video_to_array(video_path, target_size, num_frames)
    video_tensor = torch.tensor(resized_frames, dtype=torch.float32)

    # Rearrange the tensor dimensions to (batch, channel, time, height, width)
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)

    # assert video_tensor.shape == (1, 3, 16, 224, 224) #UNCOMMENT

    return video_tensor

def video_to_array(video_path, target_size=(224, 224), num_frames=16):
    '''
    Converts a given video mp4 file to a PyTorch tensor
    NOT normalized
    '''
    # Read the video
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the sampling rate to get the target number of frames

    sampling_rate = total_frames // num_frames

    if sampling_rate == 0:
        print("Got total frames: ", total_frames, "for video", video_path, "which causes division by 0 for sampling rate.")
        exit()

    # Initialize an empty list to store the resized frames
    resized_frames = []

    frame_count = 0
    sampled_count = 0

    # Loop through the video frames
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % sampling_rate == 0:
            # Resize the frame
            resized_frame = cv2.resize(frame, target_size)

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Append the resized frame to the list
            resized_frames.append(rgb_frame)
            sampled_count += 1

            if sampled_count >= num_frames:
                break

        frame_count += 1

    # Release the video capture object
    video.release()

    # Convert the list of frames to a PyTorch tensor
    resized_frames = np.array(resized_frames)

    return resized_frames

def check_folder_equality(str1, str2):
    n = len(str2)
    return str1[-n:] == str2

def image_to_tensor(image_path, target_shape=(1, 3, 1, 224, 224)):
    '''
    Returns a tensor of shape (1, 3, 1, 224, 224) from an image
    NOT normalized
    '''
    convert_tensor = transforms.ToTensor()
    img = Image.open(image_path)
    img = img.resize((target_shape[-1], target_shape[-2]), Image.ANTIALIAS)  # Resize to (width, height)

    img = convert_tensor(img)
    img = img.unsqueeze(1)  # Add the num_frames dimension
    img = img.unsqueeze(0)  # Add the batch_size dimension

    assert img.shape == target_shape
    return img.float()

def uint8_to_normalized(tensor):
    """
    Convert a uint8 tensor to a float tensor and normalize the values.
    tensor: PyTorch tensor, the uint8 tensor to convert
    """
    return utils.tensor_normalize(tensor)

def normalized_to_uint8(tensor):
    output = utils.revert_tensor_normalize(tensor)
    output = torch.round(output)
    return output

def get_test_model_input(file: str = None, data_dir: str = None):
    # First check if direct file exists
    if file:
        if file.endswith(".mp4"):
            video_tensor = video_to_tensor(file)
            video_tensor = uint8_to_normalized(video_tensor)
            return video_tensor.cuda()
        elif file.endswith(".png"):
            image_tensor = image_to_tensor(file, (1, 3, 1, 224, 224))
            image_tensor = uint8_to_normalized(image_tensor)
            return image_tensor.cuda()
        raise NotImplementedError

    random_file_from_data_dir = get_random_file(data_dir)
    return get_test_model_input(file=random_file_from_data_dir)

def spatial_sample_test_video(test_model_input):
    spatial_idx = 1
    test_model_input = utils.spatial_sampling(
        test_model_input.squeeze(0),
        spatial_idx=spatial_idx,
        min_scale=256,
        max_scale=256,
        crop_size=224,
        random_horizontal_flip=False
    )
    test_model_input = test_model_input.unsqueeze(0)
    return test_model_input

def reconstruct(mask, ground_truth, test_model_output):
    expanded_mask = mask.unsqueeze(-1).expand_as(ground_truth)
    result = torch.where(expanded_mask == 1, test_model_output, ground_truth)
    return result

def decode_raw_prediction(mask, model, num_patches, orig_image, y, mae_image=False):
    if mae_image:
        # print('orig_image.shape in decode_raw_prediction: ', orig_image.shape)
        N, T, _, H, W = orig_image.shape
        # print("y.shape in mae_image: ", y.shape)
        y = torch.reshape(y, [-1, 196])
        # print("y.shape in mae_image after reshape: ", y.shape)
        y = model.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
        # print("y.shape after vae.quantize: ", y.shape)
        y = model.vae.decode(y)
        # plt.figure(); plt.imshow(y[0].permute(1,2,0)); plt.show()
        y = F.interpolate(y, size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)
        y = torch.clip(y * 255, 0, 255).int().detach().cpu()
        # visualize the mask
        
        
        # mask = mask.permute(2, 0, 1, 3, 4)
        # mask = mask.flatten(0, 1)

        # mask = mask.permute(0, 2, 3, 1).detach().cpu()
        
        # print("mask.shape before everything: ", mask.shape)
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
        # print("mask.shape after unsqueeze: ", mask.shape)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        # print("mask.shape after unpatchify: ", mask.shape)
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        # print('orig_image.shape in decode_raw_prediction before permute flatten: ', orig_image.shape)
        # ([1, 3, 16, 224, 224])
        orig_image = orig_image.permute(2, 0, 1, 3, 4)
        orig_image = orig_image.flatten(0, 1)
        orig_image = orig_image.permute(0, 2, 3, 1).detach().cpu()
        # print('orig_image.shape in decode_raw_prediction after permute and flatten: ', orig_image.shape)
        
        # orig_image = torch.einsum('nchw->nhwc', orig_image)
        mean = constants.mean.cpu().detach()
        std = constants.std.cpu().detach()
        orig_image = (
            torch.clip((orig_image.cpu().detach() * std + mean) * 255, 0, 255).int()) #.unsqueeze(0)
        # print('orig_image.shape after unsqueeze: ', orig_image.shape)
        # MAE reconstruction pasted with visible patches
        im_paste = orig_image * (1 - mask) + y * mask
        
        return im_paste, mask, orig_image
    
    N = orig_image.shape[0]
    T = orig_image.shape[2]

    if T == 1: #Image
        repeat = model.patch_embed.t_patch_size
        orig_image = orig_image.repeat(1, 1, repeat, 1, 1)
        T = repeat
    
    # print("y.shape: ", y.shape)
    y = torch.reshape(y, [N * T, 196])

    if type(model) is torch.nn.parallel.DistributedDataParallel:
        model = model.module

    # print("y.shape before vae.quantize: ", y.shape)
    y = model.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
    # print("y.shape after vae.quantize in regular: ", y.shape)
    y = model.vae.decode(y)
    y = F.interpolate(y, size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)
    y = torch.clip(y * 255, 0, 255).int().detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)

    #patchify to get self.patch_info
    _ = model.patchify(orig_image)

    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

    orig_image = orig_image.permute(2, 0, 1, 3, 4)
    orig_image = orig_image.flatten(0, 1)
    mask = mask.permute(2, 0, 1, 3, 4)
    mask = mask.flatten(0, 1)

    mask = mask.permute(0, 2, 3, 1).detach().cpu()
    orig_image = orig_image.permute(0, 2, 3, 1).detach().cpu()
    mean = constants.mean.cpu().detach()
    std = constants.std.cpu().detach()

    orig_image = (
            torch.clip((orig_image.cpu().detach() * std + mean) * 255, 0, 255).int()).unsqueeze(0)

    # MAE reconstruction pasted with visible patches
    im_paste = orig_image * (1 - mask) + y * mask
    return im_paste, mask, orig_image

@torch.no_grad()
def visualize_prompting(model, test_cases_folder, mae_image=False, mask_ratio_image=0.75, mask_ratio_video=0.9):
    visualize_image_prompting(model, os.path.join(test_cases_folder, "test_images/"), mae_image=mae_image)
    visualize_video_prompting(model, os.path.join(test_cases_folder, "random_masked_videos/"), "random", mae_image=mae_image, mask_ratio_video=mask_ratio_video)
    # visualize_video_prompting(model, os.path.join(test_cases_folder, "temporally_masked_videos/"), "temporal", mae_image=mae_image)
    # visualize_video_prompting(model, os.path.join(test_cases_folder, "spatiotemporally_masked_1_video/"), "spatiotemporal", mae_image=mae_image)
    # visualize_video_prompting(model, os.path.join(test_cases_folder, "spatiotemporally_masked_2_videos/"), "spatiotemporal", mae_image=mae_image)
    
    # objectron_videos_path = '/shared/dannyt123/video_inpainting/test_videos/Objectron'
    # visualize_video_prompting(model, objectron_videos_path, "frame prediction", mae_image=mae_image)
    visualize_video_prompting(model, os.path.join(test_cases_folder, "davis_2x2_prompt_visualization/"), "2x2 tube", mae_image=mae_image)
    # visualize_video_prompting(model, objectron_videos_path, "frame interpolation", mae_image=mae_image)
    # visualize_video_prompting(model, objectron_videos_path, "central inpainting", mae_image=mae_image)
    # visualize_video_prompting(model, objectron_videos_path, "dynamic inpainting", mae_image=mae_image)
    
    # visualize_video_prompting(model, os.path.join(test_cases_folder, "view_videos/"), "view") # TODO

@torch.no_grad()

def visualize_image_prompting(model, input_image_viz_dir, mae_image=False):
    ### Test on images

    '''
    Masks out the bottom right quadrant of an image and inpaint it.
    '''

    for i, img_file in enumerate(os.listdir(input_image_viz_dir)):
        img_file = os.path.join(input_image_viz_dir, img_file)
        test_model_input = get_test_model_input(file=img_file)
        test_model_input = test_model_input.cuda()

        if type(model) is torch.nn.parallel.DistributedDataParallel:
            model = model.module

        # print('test_model_input.shape in image: ', test_model_input.shape)
        T = test_model_input.shape[2]
        _, test_model_output, mask = model(test_model_input, test_image=True)

        num_patches = 14
        N = test_model_input.shape[0]

        # print("test_model_output.shape in image: ", test_model_output.shape)
        test_model_output = test_model_output.view(N, T, 196, -1, 1024)
        test_model_output = test_model_output.permute(0, 1, 3, 2, 4)
        test_model_output = test_model_output.flatten(1, 2)
        test_model_output = test_model_output.flatten(1, 2)
    
        y = test_model_output.argmax(dim=-1)
        im_paste, _, _ = decode_raw_prediction(mask, model, num_patches, test_model_input, y, mae_image=mae_image)
        im_paste = im_paste.squeeze()
        im_paste = (im_paste.cpu().numpy()).astype(np.uint8)
        if im_paste.shape[0] > 1 and len(im_paste.shape) >= 4:
            im_paste = im_paste[0]
            
        # print('im_paste.shape in image: ', im_paste.shape)

        img_file = os.path.basename(os.path.normpath(img_file))
        output_img_name = str(img_file)

        image = wandb.Image(im_paste)
        wandb.log({output_img_name: image})

@torch.no_grad()
def visualize_video_prompting(model, input_video_viz_dir, test_type="", mae_image=False, mask_ratio_video=0.9):    
    if type(model) is torch.nn.parallel.DistributedDataParallel:
        model = model.module

    test_model_input = get_test_model_input(data_dir=input_video_viz_dir)
    test_model_input = spatial_sample_test_video(test_model_input)

    print("prompting video with", input_video_viz_dir)
    
    # print('test_model_input.shape in video: ', test_model_input.shape)

    if test_type == "random":
        _, test_model_output, mask = model(test_model_input, mask_ratio_video=mask_ratio_video)
    elif test_type:
        _, test_model_output, mask = model(test_model_input, video_test_type=test_type)
    else:
        raise ValueError("Invalid input_video_viz_dir")
    
    
    if mae_image:
        num_patches = 14
        y = test_model_output.argmax(dim=-1)
        im_paste, _, orig_video = decode_raw_prediction(mask, model, num_patches, test_model_input, y, mae_image)
        
        im_paste = im_paste.squeeze()
        im_paste = (im_paste.cpu().numpy()).astype(np.uint8)
        
        orig_video = orig_video.permute(0, 3, 1, 2)
        im_paste = im_paste.transpose(0, 3, 1, 2)
    else:
        im_paste, orig_video = video_generation(model, mask, test_model_input, test_model_output)
        

    folder_name = os.path.basename(os.path.normpath(input_video_viz_dir))
    video_title = "{type}_{test_type}_{folder_name}"
    input_video_title = video_title.format(type="input", test_type=test_type, folder_name=folder_name)
    output_video_title = video_title.format(type="output", test_type=test_type, folder_name=folder_name)
    
    wandb_video_object = wandb.Video(
        data_or_path=orig_video,
        fps=4, 
        format="mp4"
    )
    wandb.log({input_video_title: wandb_video_object}) 
    
    wandb_video_object = wandb.Video(
        data_or_path=im_paste,
        fps=4, 
        format="mp4"
    )
    wandb.log({output_video_title: wandb_video_object})
    
def video_generation(model, mask, test_model_input, test_model_output):
    num_patches = 14
    N = test_model_input.shape[0]

    test_model_output = test_model_output.view(N, -1, 196, 2, 1024)
    test_model_output = test_model_output.permute(0, 1, 3, 2, 4)
    test_model_output = test_model_output.flatten(1, 2)
    test_model_output = test_model_output.flatten(1, 2)

    y = test_model_output.argmax(dim=-1)
    im_paste, _, orig_video = decode_raw_prediction(mask, model, num_patches, test_model_input, y)

    im_paste = im_paste.permute((0, 1, 4, 2, 3))
    orig_video = orig_video.permute((0, 1, 4, 2, 3))
    im_paste = (im_paste.cpu().numpy()).astype(np.uint8)
    orig_video = (orig_video.cpu().numpy()).astype(np.uint8)

    return im_paste, orig_video