import torch
import numpy as np
from torchvision import datasets
import os

from torchvision.transforms import transforms


def get_image_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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
        # TODO: add kinetics and more.
        raise NotImplementedError()

    else:
        raise ValueError("Wrong dataset type.")

    return dataset_train


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, image_dataset_list, image_dataset_conf, video_dataset_list, video_dataset_conf,
                 image_pct):

        self.image_datasets = [get_dataset(ds_name, root_path, 'image') for ds_name in image_dataset_list]
        self.video_datasets = [get_dataset(ds_name, root_path, 'video') for ds_name in video_dataset_list]
        self.image_dataset_conf = np.array(image_dataset_conf) / np.sum(image_dataset_conf)
        self.video_dataset_conf = np.array(video_dataset_conf) / np.sum(video_dataset_conf)
        self.image_pct = image_pct

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, index: int):

        pct = np.random.uniform()
        if pct < self.image_pct:
            ds = np.random.choice(self.image_datasets, p=self.image_dataset_conf)
        else:
            ds = np.random.choice(self.video_datasets, p=self.video_dataset_conf)
        idx = np.random.randint(0, len(ds))
        return ds[idx]


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
