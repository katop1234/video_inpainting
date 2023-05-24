import torch
import numpy as np
from torchvision import datasets
import os
from torchvision.transforms import transforms
from util.decoder import constants
from util.kinetics import Kinetics

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
            dataset_train = Kinetics(
                mode="pretrain",
                path_to_data_dir=os.path.join(root_path, 'kinetics/train_256/'),
                sampling_rate=4,
                num_frames=16,
                train_jitter_scales=(256, 320),
                repeat_aug=1, # TODO this has been hardcoded, figure out how to make it more elegant
                jitter_aspect_relative=[0.75, 1.3333],
                jitter_scales_relative=[0.5, 1.0],
                )
        elif name == "Objectron":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/Objectron")
        elif name == "CrossTask":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/CrossTask")
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
