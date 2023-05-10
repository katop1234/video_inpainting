import torch
import wandb
import os
import numpy as np
import models_mae
import cv2
from PIL import Image
from util.decoder import utils
import random
import torch.nn.functional as F #added
from torchvision import transforms #added

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

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
    video_tensor = torch.tensor(resized_frames, dtype=torch.float32)

    # Rearrange the tensor dimensions to (batch, channel, time, height, width)
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)

    assert video_tensor.shape == (1, 3, 16, 224, 224)

    return video_tensor


def check_folder_equality(str1, str2):
    n = len(str2)
    return str1[-n:] == str2


def image_to_tensor(image_path, target_shape=(1, 3, 1, 224, 224)):
    '''
    Returns a tensor of shape (1, 3, 1, 224, 224) from an image
    NOT normalized
    '''
    convert_tensor = transforms.ToTensor() #added
    img = Image.open(image_path)
    img = img.resize((target_shape[-1], target_shape[-2]), Image.ANTIALIAS)  # Resize to (width, height)
    
    #added
    img = convert_tensor(img)

    #original 
    # img = img.resize((target_shape[-1], target_shape[-2]), Image.ANTIALIAS)  # Resize to (width, height)
    # img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Convert to a tensor and rearrange dimensions
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
            # image_tensor = uint8_to_normalized(image_tensor)
            return image_tensor.cuda()
        raise NotImplementedError

    # TODO also feed in "test_cases/final_spatiotemporal_videos/"
    if check_folder_equality(data_dir, "test_cases/final_temporal_videos/"):
        random_mp4 = get_random_file(data_dir)
        return get_test_model_input(file=random_mp4)

    elif check_folder_equality(data_dir, "test_cases/visual_prompting_images/"):
        random_png = get_random_file(data_dir)
        return get_test_model_input(file=random_png)

    raise NotImplementedError


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
    return test_model_input  # shape (1, 3, 16, 224, 224)


def reconstruct(mask, ground_truth, test_model_output):
    # print("ground_truth.shape: ", ground_truth.shape)
    # print("test_model_output.shape: ", test_model_output.shape)
    # print("mask.shape: ", mask.shape)
    expanded_mask = mask.unsqueeze(-1).expand_as(ground_truth)
    result = torch.where(expanded_mask == 1, test_model_output, ground_truth)
    return result

#added function
def decode_raw_prediction(mask, model, num_patches, orig_image, y):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    
    T = orig_image.shape[2]
    if T == 16:
        y = torch.reshape(y, [16, 196])
    
    if type(model) is torch.nn.parallel.DistributedDataParallel:
        y = model.module.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
        y = model.module.vae.decode(y)
        y = F.interpolate(y, size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)
        y = torch.clip(y * 255, 0, 255).int().detach().cpu()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_embed.patch_size[0] ** 2 * 3)

        #patchify to get self.patch_info 
        _ = model.module.patchify(orig_image)

        mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
    elif type(model) is models_mae.MaskedAutoencoderViT:
        y = model.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
        y = model.vae.decode(y)
        y = F.interpolate(y, size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)
        y = torch.clip(y * 255, 0, 255).int().detach().cpu()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)

        #patchify to get self.patch_info 
        _ = model.patchify(orig_image)

        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    else:
        raise NotImplementedError("Something's funky")

    
    if T == 1: #image
        orig_image = torch.squeeze(orig_image, 2)
        mask = torch.squeeze(mask, 2)
    elif T == 16: #video
        orig_image = orig_image.permute(2, 1, 0, 3, 4)
        # orig_image = torch.einsum('ncthw->tcnhw', orig_image) #for video we treat the Time as the Batch [1, 3, 16, 224, 224]->[16, 3, 1, 224, 224]
        orig_image = torch.squeeze(orig_image, 2) #[16, 3, 224, 224]

        mask = mask.permute(2, 1, 0, 3, 4)
        # mask = torch.einsum('ncthw->tcnhw', mask) #same as orig_image
        mask = torch.squeeze(mask, 2) #same as orig_image
    else: 
        raise NotImplementedError("Not video or image")

    mask = mask.permute(0, 2, 3, 1).detach().cpu()
    orig_image = orig_image.permute(0, 2, 3, 1).detach().cpu()
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    # orig_image = torch.einsum('nchw->nhwc', orig_image).cpu()
    print("mask: ", mask)
    imagenet_mean = imagenet_mean.cpu().detach()
    imagenet_std = imagenet_std.cpu().detach()
    print("orig_image.shape: ", orig_image.shape)
    imagenet_mean = torch.tensor([0, 0, 0])
    imagenet_std = torch.tensor([1, 1, 1])
    orig_image = (
        torch.clip((orig_image[0].cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int()).unsqueeze(0) #have to change/might not want to

    # orig_image = normalized_to_uint8(orig_image)

    # orig_image = (
    #     torch.clip((orig_image[0].cpu().detach()), 0, 255).int()).unsqueeze(0) #have to change/might not want to
    print("orig_image.shape: ", orig_image.shape)
    # MAE reconstruction pasted with visible patches
    im_paste = orig_image * (1 - mask) + y * mask
    # im_paste = orig_image
    return im_paste, mask, orig_image


@torch.no_grad()
def visualize_prompting(model, input_image_viz_dir, input_video_viz_dir):
    model.eval()
    visualize_image_prompting(model, input_image_viz_dir)
    visualize_video_prompting(model, input_video_viz_dir)
    # model.train()

@torch.no_grad()
def visualize_image_prompting(model, input_image_viz_dir):
    ### Test on images
    for i, img_file in enumerate(os.listdir(input_image_viz_dir)):
        img_file = os.path.join(input_image_viz_dir, img_file)
        test_model_input = get_test_model_input(file=img_file) #added comment [1, 3, 1, 224, 224]
        test_model_input = test_model_input.cuda()

        with torch.no_grad():
            if type(model) is torch.nn.parallel.DistributedDataParallel:
                _, test_model_output, mask = model.module(test_model_input, test_image=True)
            elif type(model) is models_mae.MaskedAutoencoderViT:
                _, test_model_output, mask = model(test_model_input, test_image=True) #running forward logically same as generate_raw_prediction #might need to make if statement
            else: 
                raise NotImplementedError("Something's funky")

        num_patches = 14
        y = test_model_output.argmax(dim=-1) #should still work
        im_paste, mask, orig_image = decode_raw_prediction(mask, model, num_patches, test_model_input, y) #check
        im_paste = im_paste.unsqueeze(0)
        im_paste = im_paste.permute(0, 4, 1, 2, 3)
        # im_paste = torch.einsum('nthwc->ncthw', im_paste)
        
        test_model_output = im_paste
        print("test_model_output.shape: ", test_model_output.shape)
        test_model_output = test_model_output.squeeze(2)
        print("test_model_output.shape: ", test_model_output.shape)
        denormalized_img = test_model_output.permute(0, 2, 3, 1)   # (224, 224, 3)

        # test_model_output = normalized_to_uint8(im_paste)
        # denormalized_img = test_model_output.squeeze(0).permute(1, 2, 3, 0) #.squeeze(0)  # (224, 224, 3)

        image_array = (denormalized_img.cpu().numpy()).astype(np.uint8)
        # print("image_array.shape: ", image_array.shape)
        # return image_array, orig_image
        
        output_img_name = 'test_model_output_img' + str(i) + '.png'
        image = wandb.Image(image_array)
        wandb.log({output_img_name: image}) #temporarily commented out for jupyter playing around


@torch.no_grad()
def visualize_video_prompting(model, input_video_viz_dir="test_cases/final_temporal_videos/"):
    test_model_input = get_test_model_input(data_dir=input_video_viz_dir) #[1, 3, 16, 224, 224]
    test_model_input = spatial_sample_test_video(test_model_input) #[1, 3, 16, 224, 224]

    with torch.no_grad():
        # TODO change test_temporal to True later when it works
        with torch.no_grad():
            if type(model) is torch.nn.parallel.DistributedDataParallel:
                _, test_model_output, mask = model.module(test_model_input)
            elif type(model) is models_mae.MaskedAutoencoderViT:
                _, test_model_output, mask = model(test_model_input)
            else:
                raise NotImplementedError("Something's funky")

    num_patches = 14
    y = test_model_output.argmax(dim=-1)
    im_paste, mask, orig_image = decode_raw_prediction(mask, model, num_patches, test_model_input, y)
    im_paste = im_paste.unsqueeze(0)
    im_paste = im_paste.permute((0, 1, 4, 2, 3))

    wandb_video_object = wandb.Video(
        # data_or_path=test_model_output_np,
        data_or_path=im_paste,
        fps=30
    )
    wandb.log({"video": wandb_video_object})
