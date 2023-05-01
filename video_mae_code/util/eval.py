import torch
import wandb

@torch.no_grad()
def visualize_prompting(model, input_image_viz_dir, input_video_viz_dir):
    model.eval()
    visualize_image_prompting(model, input_image_viz_dir)
    visualize_video_prompting(model, input_video_viz_dir)
    model.train()

@torch.no_grad()
def visualize_image_prompting(model, input_image_viz_dir):
    return

@torch.no_grad()
def visualize_video_prompting(model, input_video_viz_dir):
    return




### Starting evaluation
# model.eval()
#
# ### Test on video
# test_model_input = get_test_model_input(data_dir="test_cases/final_temporal_videos/") # DEBUG check range and shape at each step
#
# save_frames_as_mp4(normalized_to_uint8(test_model_input), file_name="test_input_video.mp4")
#
# test_model_input = spatial_sample_test_video(test_model_input)
#
# # Override and just get a video directly from training dataset
# # test_model_input = get_test_model_input_from_kinetics(dataset_train) # NOTE this relies on using Kinetics and CVF as original dataset_train
#
# with torch.no_grad():
#     # TODO change test_temporal to True later
#     _, test_model_output, _ = model(test_model_input)
#
# if type(model) is torch.nn.parallel.DistributedDataParallel:
#     test_model_output = model.module.unpatchify(test_model_output)
# elif type(model) is models_mae.MaskedAutoencoderViT:
#     test_model_output = model.unpatchify(test_model_output)
# else:
#     raise NotImplementedError("Something's funky")
#
# test_model_output = normalized_to_uint8(test_model_output)
#
# save_test_output(test_model_output, name="test_output_video.mp4")
#
# if not (test_image or test_video) and misc.is_main_process():
#     wandb_video_object = wandb.Video(
#         data_or_path= "test_output_video.mp4",
#         caption=epoch,
#         fps=30,
#         )
#     wandb.log({"video": wandb_video_object})
#
# ### Test on images
# test_img_folder = "test_cases/visual_prompting_images/"
# for i, img_file in enumerate(os.listdir(test_img_folder)):
#     img_file = test_img_folder + img_file # TODO follow the dimensionality of the image
#     test_model_input = get_test_model_input(file=img_file)
#     test_model_input = test_model_input.cuda()
#
#     with torch.no_grad():
#         # TODO why does it mask it weirdly, like only the bottom 1/4 even after i fixed the mask_test_image function??
#         _, test_model_output, _ = model(test_model_input, test_image=True)
#
#     if type(model) is torch.nn.parallel.DistributedDataParallel:
#         test_model_output = model.module.unpatchify(test_model_output)
#     elif type(model) is models_mae.MaskedAutoencoderViT:
#         test_model_output = model.unpatchify(test_model_output)
#     else:
#         raise NotImplementedError("Something's funky")
#
#     # TODO USE SINGLE MEAN STD FOR IMG + VIDEO, see wherever normalized_to_uint8 and inverse is called
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#
#     test_model_output = normalized_to_uint8(test_model_output, mean, std)
#
#     # Rearrange dimensions to have channels last, and remove unnecessary dimensions
#     denormalized_img = test_model_output.squeeze(0).permute(1, 2, 3, 0).squeeze(0) # (224, 224, 3)
#
#     # Convert to numpy array, scale back to [0, 255] and convert to uint8 data type
#     image_array = (denormalized_img.cpu().numpy()).astype(np.uint8)
#
#     output_img_name = 'test_model_output_img' + str(i) + '.png'
#
#     save_test_output(image_array, output_img_name)
#
#     if not (test_image or test_video) and misc.is_main_process():
#         image = wandb.Image(
#             image_array,
#             )
#
#         wandb.log({output_img_name: image})
#
# model.train()
# exit() #