from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from util.decoder import constants
from util.kinetics import Kinetics
from torchvision import datasets
from torchvision.transforms import transforms
import glob
import numpy as np
import os
import torch

class VideoDataset(Kinetics):
    def __init__(self, path_to_data_dir, 
                 mode="pretrain", 
                 sampling_rate=4, 
                 num_frames=16, 
                 train_jitter_scales=(256, 320), 
                 repeat_aug=1, 
                 jitter_aspect_relative=[0.75, 1.3333], 
                 jitter_scales_relative=[0.5, 1.0]):
        
        super().__init__(path_to_data_dir, 
                         mode=mode, 
                         sampling_rate=sampling_rate, 
                         num_frames=num_frames, 
                         train_jitter_scales=train_jitter_scales, 
                         repeat_aug=repeat_aug, 
                         jitter_aspect_relative=jitter_aspect_relative, 
                         jitter_scales_relative=jitter_scales_relative)

class AtariDataset(Dataset):
    def __init__(self, path_to_data_dir):
        self.root_dir = path_to_data_dir
        self.transform = torch.nn.Sequential(
            Resize((224, 224)),
            ToTensor(),
        )
        self.games = ['mspacman', 'pinball', 'qbert', 'revenge', 'spaceinvaders']
        self.subfolders = []

        # Populate the list of subfolders
        for game in self.games:
            game_folder = os.path.join(self.root_dir, game)
            subfolders = glob.glob(os.path.join(game_folder, '*'))
            self.subfolders.extend(subfolders)

    def __len__(self):
        # Return the total number of sub-subfolders across all games
        return len(self.subfolders)

    def __getitem__(self, idx):
        # Get the list of image paths in the selected subfolder
        image_paths = sorted(glob.glob(os.path.join(self.subfolders[idx], '*.png')))

        # Ensure there are at least 16 images in the subfolder
        if len(image_paths) < 16:
            raise ValueError(f"Found a subfolder with less than 16 images: {self.subfolders[idx]}")

        # Select a random start index for the sequence of 16 frames
        start_idx = torch.randint(0, len(image_paths) - 15, (1,)).item()

        # Load 16 consecutive images starting from the selected index and apply the transform
        images = [self.transform(Image.open(image_paths[i])) for i in range(start_idx, start_idx + 16)]

        # Stack the images into a tensor of shape [1, 3, 16, 224, 224]
        images = torch.stack(images)
        images = images.unsqueeze(0)

        return images

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
            dataset_train = datasets.ImageFolder(os.path.join(root_path, 'ilsvrc/train'), transform=transforms_train)
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
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/CrossTask")
        elif name == 'atari':
            dataset_train = AtariDataset(path_to_data_dir="/shared/katop1234/Datasets/atari")
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
        print("getting output from getitem", "index", index, "len(self.datasets)", len(self.datasets))
        sampled_ds_index = np.random.choice(np.arange(0, len(self.datasets)), p=self.conf)
        print("got sampled ds index", sampled_ds_index, index)
        ds = self.datasets[sampled_ds_index]
        print("ds", ds, len(ds), index)
        output_index = np.random.randint(0, len(ds))
        print("got output index", output_index)
        output = ds[output_index]
        print("output", output[0].shape, index)
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
