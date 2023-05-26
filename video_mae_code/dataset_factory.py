import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from video_mae_code.util.decoder import constants
from video_mae_code.util.kinetics import Kinetics
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms import Compose, Resize, ToTensor
import glob
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

    def _construct_loader(self):
        """
        Overwrite kinetics loader variables
        """
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        # List all video files in the path_to_data_dir
        for filename in os.listdir(self._path_to_data_dir):
            if filename.endswith(".mp4"):  # assuming the videos are mp4 format
                self._path_to_videos.append(os.path.join(self._path_to_data_dir, filename))
                self._labels.append(0)  # append 0 as label for all videos
                self._spatial_temporal_idx.append(0)  # append 0 as spatial_temporal_idx for all videos
        
        for i in range(len(self._path_to_videos)):
            self._video_meta[i] = {}

def get_image_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=constants.mean, std=constants.std)])

def get_dataset(name, root_path, ds_type):
    if ds_type == 'image':
        transforms_train = get_image_transforms()
        if name == 'imagenet':
            dataset_train = datasets.ImageFolder(os.path.join(root_path, 'ilsvrc/train'), 
                                                 transform=transforms_train)
        elif name == 'cvf':
            dataset_train = datasets.ImageFolder(os.path.join(root_path, 'arxiv_resized_train_val_split/train/'),
                                                 transform=transforms_train)
        else:
            raise ValueError("Wrong dataset name.")

    elif ds_type == 'video':
        if name == "kinetics":
            dataset_train = VideoDataset(path_to_data_dir=os.path.join(root_path, 'kinetics/train_256/'))
        elif name == "Objectron":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/Objectron")
        elif name == "CrossTask":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/CrossTask_vids")
        elif name == 'atari':
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/atari_mp4s/")
        elif name == "SSV2":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/SSV2_videos/") 
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Wrong dataset type.")

    return dataset_train

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, image_dataset_list, image_dataset_conf, video_dataset_list, video_dataset_conf,
                 image_pct):
        
        image_pct = float(image_pct)
        image_dataset_conf = [float(x) for x in image_dataset_conf]
        video_dataset_conf = [float(x) for x in video_dataset_conf]
        
        image_datasets = [get_dataset(ds_name, root_path, 'image') for ds_name in image_dataset_list]
        video_datasets = [get_dataset(ds_name, root_path, 'video') for ds_name in video_dataset_list]
        datasets = image_datasets + video_datasets
 
        conf = list(image_pct * np.array(image_dataset_conf)) + list((1 - image_pct) * np.array(video_dataset_conf))
        conf = [i / sum(conf) for i in conf]
        self.datasets = datasets
        self.conf = conf

    def __len__(self):
        # return 16 # For testing purposes
        return 79000

    def __getitem__(self, index: int):
        sampled_ds_index = np.random.choice(np.arange(0, len(self.datasets)), p=self.conf)
        ds = self.datasets[sampled_ds_index]
        output_index = np.random.randint(0, len(ds))
        output = ds[output_index]
        return output

if __name__ == '__main__':
    root_path = ''
    image_dataset_list = ['imagenet', 'cvf']
    image_dataset_conf = [0.5, 0.5]
    video_dataset_list = ['kinetics400', 'cityscapes']
    video_dataset_conf = [1]
    image_pct = 0.5
    ds = MergedDataset(root_path, image_dataset_list, image_dataset_conf, video_dataset_list, video_dataset_conf,
                       image_pct)

    for i in range(100):
        x, y = ds[i]
        print(x.shape, y)
