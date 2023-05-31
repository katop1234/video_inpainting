import os
from torch.utils.data import Dataset
from util.decoder import constants
from util.kinetics import Kinetics
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import torch

class VideoDataset(Kinetics):
    def __init__(self, path_to_data_dir):
        super().__init__(path_to_data_dir=path_to_data_dir,
                         mode="pretrain",
                         sampling_rate=4,
                         num_frames=16,
                         train_jitter_scales=(256, 320),
                         repeat_aug=1,
                         jitter_aspect_relative=[0.75, 1.3333],
                         jitter_scales_relative=[0.5, 1.0])

def get_image_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=constants.mean, std=constants.std)])

def get_dataset(name, root_path, ds_type):
    if ds_type == 'image':
        transforms_train = get_image_transforms()
        if name == 'cvf':
            dataset_train = datasets.ImageFolder(os.path.join(root_path, 'arxiv_resized_train_val_split/train/'),
                                                 transform=transforms_train)
        elif name == 'imagenet':
            dataset_train = datasets.ImageFolder(os.path.join(root_path, 'ilsvrc/train'), 
                                                 transform=transforms_train)
        else:
            raise ValueError("Wrong dataset name.")

    elif ds_type == 'video':
        if name == 'atari':
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/atari_mp4s/")
        elif name == "CrossTask":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/CrossTask_vids")
        elif name == "kinetics":
            dataset_train = VideoDataset(path_to_data_dir=os.path.join(root_path, 'kinetics/train_256/'))
        elif name == "Objectron":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/Objectron")
        elif name == "SSV2":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/SSV2_videos/") 
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Wrong dataset type.")

    return dataset_train


def combined_gen(image_gen, video_gen, accum_iter_img, accum_iter_vid, image_video_ratio, num_iter):
    i = 0
    while i <= num_iter:
        if np.random.random() > image_video_ratio:
            gen = image_gen
            accum_iter = accum_iter_img
        else:
            gen = video_gen
            accum_iter = accum_iter_vid

        while accum_iter >= 1 and i <= num_iter:
            sample = next(gen)
            yield sample, accum_iter
            accum_iter -= 1
            i += 1
    return


class CombinedGen:
    def __init__(self, image_gen, video_gen, accum_iter_img, accum_iter_vid, image_video_ratio):
        self.image_gen = iter(image_gen)
        self.video_gen = iter(video_gen)
        self.accum_iter_img = accum_iter_img
        self.accum_iter_vid = accum_iter_vid
        self.image_video_ratio = image_video_ratio

    def __iter__(self):
        return combined_gen(self.image_gen, self.video_gen, self.accum_iter_img, self.accum_iter_vid, self.image_video_ratio, len(self))

    def __len__(self):
        return len(self.image_gen) # assuming we use image len for now


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, image_dataset_list, image_dataset_conf, ds_type):
        image_dataset_conf = [float(x) for x in image_dataset_conf]
        image_datasets = [get_dataset(ds_name, root_path, ds_type) for ds_name in image_dataset_list]
        conf = [i / sum(image_dataset_conf) for i in image_dataset_conf]
        self.datasets = image_datasets
        self.conf = conf

    def __len__(self):
        return 79000

    def __getitem__(self, index: int):
        sampled_ds_index = np.random.choice(np.arange(0, len(self.datasets)), p=self.conf)
        ds = self.datasets[sampled_ds_index]
        output_index = np.random.randint(0, len(ds))
        output = ds[output_index]
        return output