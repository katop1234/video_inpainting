# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import imageio

import torch.utils.data
from PIL import Image
import matplotlib

from iopath.common.file_io import g_pathmgr as pathmgr
from util.decoder.decoder import get_start_end_idx, temporal_sampling
from torchvision import transforms

from util.decoder import decoder as decoder, utils as utils, video_container as container
from util.decoder.random_erasing import RandomErasing
from util.decoder.transform import create_random_augment

def tensor_is_video(tensor):
    return tensor.size()[1:] == (3, 16, 224, 224)

def tensor_is_image(tensor):
    return tensor.size()[1:] == (3, 1, 224, 224)

class KineticsAndCVF(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(
        self,
        mode,
        path_to_csv="/home/katop1234/video_inpainting/fb_mae_st/mae_st/kinetics_videos.csv", # This is the path to names of all kinetics files TODO
        path_to_data_dir="/shared/group/kinetics/train_256/", # video
        path_to_image_data_dir="/shared/amir/dataset/arxiv_resized_train_val_split/train/", # FOR IMAGES
        # decoding setting
        sampling_rate=4,
        num_frames=16,
        target_fps=30,
        # train aug settings
        train_jitter_scales=(256, 320),
        train_crop_size=224,
        train_random_horizontal_flip=True,
        # test setting, multi crops
        test_num_ensemble_views=10,
        test_num_spatial_crops=3,
        test_crop_size=256,
        # norm setting
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        # other parameters
        enable_multi_thread_decode=False,
        use_offset_sampling=True,
        inverse_uniform_sampling=False,
        num_retries=10,
        # pretrain augmentation
        repeat_aug=1,
        aa_type="rand-m7-n4-mstd0.5-inc1",
        pretrain_rand_flip=True,
        pretrain_rand_erase_prob=0.25,
        pretrain_rand_erase_mode="pixel",
        pretrain_rand_erase_count=1,
        pretrain_rand_erase_split=False,
        rand_aug=False,
        jitter_scales_relative=[0.5, 1.0],
        jitter_aspect_relative=[0.75, 1.3333],
    ):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "pretrain",
            "finetune",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.aa_type = aa_type
        self.pretrain_rand_flip = pretrain_rand_flip
        self.pretrain_rand_erase_prob = pretrain_rand_erase_prob
        self.pretrain_rand_erase_mode = pretrain_rand_erase_mode
        self.pretrain_rand_erase_count = pretrain_rand_erase_count
        self.pretrain_rand_erase_split = pretrain_rand_erase_split

        self.jitter_aspect_relative = jitter_aspect_relative
        self.jitter_scales_relative = jitter_scales_relative

        print(
            f"jitter_aspect_relative {jitter_aspect_relative} jitter_scales_relative {jitter_scales_relative}"
        )

        self._repeat_aug = repeat_aug
        self._video_meta = {}
        self._num_retries = num_retries
        self._path_to_data_dir = path_to_data_dir
        self._path_to_csv = path_to_csv
        self.path_to_image_data_dir = path_to_image_data_dir # FOR IMAGES

        self._train_jitter_scales = train_jitter_scales
        self._train_crop_size = train_crop_size
        self._train_random_horizontal_flip = train_random_horizontal_flip

        self._test_num_ensemble_views = test_num_ensemble_views
        self._test_num_spatial_crops = test_num_spatial_crops
        self._test_crop_size = test_crop_size

        self._sampling_rate = sampling_rate
        self._num_frames = num_frames
        self._target_fps = target_fps

        self._mean = mean
        self._std = std

        self._enable_multi_thread_decode = enable_multi_thread_decode
        self._inverse_uniform_sampling = inverse_uniform_sampling
        self._use_offset_sampling = use_offset_sampling

        print(self)
        print(locals())

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["pretrain", "finetune", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = test_num_ensemble_views * test_num_spatial_crops

        print("Constructing Kinetics {}...".format(mode))

        self.image_folder = path_to_image_data_dir # FOR IMAGES

        self._construct_loader()
        if self.mode in ["pretrain", "val", "test"]:
            self.rand_aug = False
            print("Perform standard augmentation")
        else:
            self.rand_aug = rand_aug
            print("Perform rand augmentation")
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        # TODO kinetics has many subfolders, which then contain the raw videos
        # TODO make sure this code can access the videos in those subfolders

        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "test",
        }
        path_to_file = os.path.join(
            self._path_to_data_dir,
            "{}.csv".format(csv_file_name[self.mode]),
        )

        path_to_file = self._path_to_csv
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2
                path, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(path))
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        print(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

        # Add support for image folder
        if self.image_folder: # FOR IMAGES
            csv_file_path = "/home/katop1234/video_inpainting/fb_mae_st/mae_st/CVF_images.csv"
            image_file_lookup = {}

            with open(csv_file_path, "r") as csv_file:
                for line in csv_file.readlines():
                    line = line.strip()
                    file_path, label = line.rsplit(",", 1)
                    file_path = file_path.strip()
                    label = int(label.strip())
                    image_file_lookup[label] = file_path

            self.image_file_lookup = image_file_lookup
            self.num_images = len(image_file_lookup)
        
        # Get total items for iter
        self.total_items = self.num_images + len(self._path_to_videos)

        # Get indices for image and video
        self.image_indices = [i for i in range(self.num_images)]
        self.video_indices = [i for i in range(self.num_images, self.num_images + len(self._path_to_videos))]

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["pretrain", "finetune", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale, max_scale = self._train_jitter_scales
            crop_size = self._train_crop_size
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self._test_num_spatial_crops
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self._test_num_spatial_crops)
                if self._test_num_spatial_crops > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self._test_crop_size] * 3
                if self._test_num_spatial_crops > 1
                else [self._train_jitter_scales[0]] * 2 + [self._test_crop_size]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        if 0 <= index < self.num_images:  # SELECTION OF IMAGE OR VIDEO
            # Randomly select an image

            img_path = self.image_file_lookup[index]

            # Transformation to apply
            output_size = (224, 224)  # h, w
            transforms_train = transforms.Compose([
                transforms.RandomResizedCrop(output_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            # Load the image from file
            image = Image.open(img_path)

            transformed_images_list = []
            for _ in range(self._repeat_aug):
                # Apply the transforms
                transformed_image = transforms_train(image)

                # Add a batch dimension
                transformed_image = transformed_image.unsqueeze(0)

                # Add a frame dimension
                transformed_image = transformed_image.unsqueeze(2)

                assert tensor_is_image(transformed_image)
                transformed_images_list.append(transformed_image)

            # Stack the transformed images
            transformed_images = torch.cat(transformed_images_list, dim=0)

            if self.mode in ["test"]:
                return transformed_images, torch.tensor(int(index)), index
            else:
                return transformed_images, torch.tensor(int(index))
        elif index >= self.num_images:

            index = index - self.num_images # Sample from the videos you have

            self.choose_image = True # select images next batch
            sampling_rate = self._sampling_rate
            # Try to decode and sample a clip from a video. If the video can not be
            # decoded, repeatly find a random video replacement that can be decoded.
            for i_try in range(self._num_retries):
                video_container = None
                video_path = self._path_to_videos[index]
                try:
                    video_container = container.get_video_container(
                        video_path,
                        self._enable_multi_thread_decode,
                    )

                except Exception as e:
                    print(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    print(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                frames, fps, decode_all_video = decoder.decode(
                    video_container,
                    sampling_rate,
                    self._num_frames,
                    temporal_sample_index,
                    self._test_num_ensemble_views,
                    video_meta=self._video_meta[index],
                    target_fps=self._target_fps,
                    max_spatial_scale=min_scale,
                    use_offset=self._use_offset_sampling,
                    rigid_decode_all_video=self.mode in ["pretrain"],
                )

                assert frames is not None, 'decoder does not work!'

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    print(
                        "Failed to decode video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                frames_list = []
                label_list = []
                label = self._labels[index]
                if self.rand_aug:
                    for i in range(self._repeat_aug):
                        clip_sz = sampling_rate * self._num_frames / self._target_fps * fps
                        start_idx, end_idx = get_start_end_idx(
                            frames.shape[0],
                            clip_sz,
                            temporal_sample_index if decode_all_video else 0,
                            self._test_num_ensemble_views if decode_all_video else 1,
                            use_offset=self._use_offset_sampling,
                        )
                        # Perform temporal sampling from the decoded video.
                        new_frames = temporal_sampling(
                            frames, start_idx, end_idx, self._num_frames
                        )
                        new_frames = self._aug_frame(
                            new_frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        frames_list.append(new_frames)
                        label_list.append(label)
                else:
                    # T H W C -> C T H W.
                    for i in range(self._repeat_aug):
                        clip_sz = sampling_rate * self._num_frames / self._target_fps * fps
                        start_idx, end_idx = get_start_end_idx(
                            frames.shape[0],
                            clip_sz,
                            temporal_sample_index if decode_all_video else 0,
                            self._test_num_ensemble_views if decode_all_video else 1,
                            use_offset=self._use_offset_sampling,
                        )
                        # Perform temporal sampling from the decoded video.
                        new_frames = temporal_sampling(
                            frames, start_idx, end_idx, self._num_frames
                        )

                        new_frames = utils.tensor_normalize(
                            new_frames, self._mean, self._std
                        )
                        new_frames = new_frames.permute(3, 0, 1, 2)

                        scl, asp = (
                            self.jitter_scales_relative,
                            self.jitter_aspect_relative,
                        )
                        relative_scales = (
                            None
                            if (self.mode not in ["pretrain", "finetune"] or len(scl) == 0)
                            else scl
                        )
                        relative_aspect = (
                            None
                            if (self.mode not in ["pretrain", "finetune"] or len(asp) == 0)
                            else asp
                        )

                        # Perform data augmentation.
                        new_frames = utils.spatial_sampling(
                            new_frames,
                            spatial_idx=spatial_sample_index,
                            min_scale=min_scale,
                            max_scale=max_scale,
                            crop_size=crop_size,
                            random_horizontal_flip=self._train_random_horizontal_flip,
                            inverse_uniform_sampling=self._inverse_uniform_sampling,
                            aspect_ratio=relative_aspect,
                            scale=relative_scales,
                        )
                        frames_list.append(new_frames)
                        label_list.append(label)
                frames = torch.stack(frames_list, dim=0)
            
                assert tensor_is_video(frames), "got " + str(frames.shape)

                if self.mode in ["test"]:
                    return frames, torch.tensor(label_list), index
                else:
                    return frames, torch.tensor(label_list)
            else:
                raise RuntimeError(
                    "Failed to fetch video after {} retries.".format(self._num_retries)
                )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.aa_type,
            interpolation="bicubic",
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames,
            (0.45, 0.45, 0.45),
            (0.225, 0.225, 0.225),
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.jitter_scales_relative,
            self.jitter_aspect_relative,
        )
        relative_scales = (
            None
            if (self.mode not in ["pretrain", "finetune"] or len(scl) == 0)
            else scl
        )
        relative_aspect = (
            None
            if (self.mode not in ["pretrain", "finetune"] or len(asp) == 0)
            else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.pretrain_rand_flip,
            inverse_uniform_sampling=False,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=False,
        )

        if self.pretrain_rand_erase_prob > 0.0:
            erase_transform = RandomErasing(
                self.pretrain_rand_erase_prob,
                mode=self.pretrain_rand_erase_mode,
                max_count=self.pretrain_rand_erase_count,
                num_splits=self.pretrain_rand_erase_count,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.total_items

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

import random
import torch

# Instantiate the Kinetics class
kinetics_dataset = KineticsAndCVF(mode="pretrain")

def save_video_tensor_as_mp4(video_tensor, output_file="sample_video.mp4", fps=30):
    if tensor_is_video(video_tensor):
        # Convert the tensor to numpy array and move the color channel to the end
        video_array = video_tensor.numpy().squeeze().transpose(0, 2, 3, 1)
        
        # Convert the video array to uint8
        video_array = video_array.astype(np.uint8)
        
        # Save the video array as an mp4 file
        with imageio.get_writer(output_file, format='FFMPEG', mode='I', fps=fps) as writer:
            for frame_num in range(16):
                frame = video_array[:, :, :, frame_num].transpose(1, 2, 0)
                writer.append_data(frame)
    else:
        print("got image")

