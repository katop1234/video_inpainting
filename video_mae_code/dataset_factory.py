import os
from torch.utils.data import Dataset
from util.decoder import constants
from util.kinetics import Kinetics
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import torch

class VideoDataset(Kinetics):
    '''
    Inherit Kinetics class and overwrite the loader variables in _construct_loader function
    since we aren't concerned with classification
    '''
    def __init__(
        self,
        path_to_data_dir,
        train_jitter_scales=(256, 320),
        train_crop_size=224,
        repeat_aug=1,
        jitter_aspect_relative=[0.75, 1.3333],
        jitter_scales_relative=[0.5, 1.0],
        train_random_horizontal_flip=False,
        pretrain_rand_flip=False,
        pretrain_rand_erase_prob=0,
        rand_aug=False,
    ):
        super().__init__(
            path_to_data_dir=path_to_data_dir,
            mode="pretrain",
            train_jitter_scales=train_jitter_scales,
            train_crop_size=train_crop_size,
            repeat_aug=repeat_aug,
            jitter_aspect_relative=jitter_aspect_relative,
            jitter_scales_relative=jitter_scales_relative,
            train_random_horizontal_flip=train_random_horizontal_flip,
            pretrain_rand_flip=pretrain_rand_flip,
            pretrain_rand_erase_prob=pretrain_rand_erase_prob,
            rand_aug=rand_aug
        )

    def _construct_loader(self):
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
    # TODO make this compatible with where the automated scripts download all the files
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
            # The video is (210, 160) and we crop the top (160, 160) deterministically
            dataset_train =  VideoDataset(
                path_to_data_dir="/shared/katop1234/Datasets/atari_mp4s_120fps/",
                train_jitter_scales=(160, 160),
                train_crop_size = 160,
                train_random_horizontal_flip=False,
                pretrain_rand_flip=False,
                pretrain_rand_erase_prob=0,
                rand_aug=False,
            )
        elif name == "CrossTask":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/CrossTask_vids/")
        elif name == "kinetics":
            dataset_train = VideoDataset(path_to_data_dir=os.path.join(root_path, 'kinetics/train_256/'))
        elif name == "Objectron":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/Objectron/")
        elif name == "SSV2":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/SSV2_videos/") 
        elif name == "UCF101":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/UCF101/") 
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Wrong dataset type.")

    return dataset_train

def combined_gen(image_itr_cls, video_itr_cls, accum_iter_img, accum_iter_vid, image_itr, video_itr, num_iter):
    i = 0
    image_gen = iter(image_itr_cls)
    video_gen = iter(video_itr_cls)

    while i < num_iter:

        for j in range(image_itr):
            accum_iter = accum_iter_img
            while accum_iter >= 1:  # allow to exceed epoch to satisfy accum_iter
                try:
                    sample = next(image_gen)
                except StopIteration as e:
                    image_gen = iter(image_itr_cls)
                    sample = next(image_gen)
                yield sample, accum_iter
                accum_iter -= 1
                i += 1

        for j in range(video_itr):
            accum_iter = accum_iter_vid
            while accum_iter >= 1:  # allow to exceed epoch to satisfy accum_iter
                try:
                    sample = next(video_gen)
                except StopIteration as e:
                    video_gen = iter(video_itr_cls)
                    sample = next(video_gen)
                yield sample, accum_iter
                accum_iter -= 1
                i += 1

def iter_gen(accum_iter, i, gen, it_cls, n):
    for j in range(n):
        accum_iter = accum_iter
        while accum_iter >= 1:  # allow to exceed epoch to satisfy accum_iter
            try:
                sample = next(gen)
            except StopIteration as e:
                gen = iter(it_cls)
                sample = next(gen)
            yield sample, accum_iter
            accum_iter -= 1
            i += 1
    return i, gen


class CombinedGen:
    def __init__(self, image_gen_cls, video_gen_cls, accum_iter_img, accum_iter_vid, image_itr, video_itr):
        self.image_gen = image_gen_cls
        self.video_gen = video_gen_cls
        self.accum_iter_img = accum_iter_img
        self.accum_iter_vid = accum_iter_vid
        self.image_itr = image_itr
        self.video_itr = video_itr
        self.num_iter_per_epoch = 24*(accum_iter_img*image_itr + accum_iter_vid*video_itr)


    def __iter__(self):
        return combined_gen(self.image_gen, self.video_gen, self.accum_iter_img, self.accum_iter_vid, self.image_itr, self.video_itr, len(self))

    def __len__(self):
        return self.num_iter_per_epoch # carefuly computed to avoid accum_iter updates here


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

def visualize_input_from_dataset(name, root_path, ds_type, index=-1):
    dataset = get_dataset(name, root_path, ds_type)

    if index == -1:
        index = np.random.randint(0, len(dataset))

    input = dataset[index][0]
    print(input.shape)
    exit()
