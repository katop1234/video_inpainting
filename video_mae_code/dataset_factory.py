import os
from torch.utils.data import Dataset
from util.decoder import constants
from util.kinetics import Kinetics
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler

class ImageNetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.categories = [dir for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir)) and not dir.startswith('.fuse_hidden')]
        self.image_paths = []
        self.image_labels = []

        for category in self.categories:
            category_path = os.path.join(root_dir, category)
            for image_name in os.listdir(category_path):
                if image_name.startswith('.fuse_hidden'):  # Skip any hidden fuse files in the image level
                    continue

                self.image_paths.append(os.path.join(category_path, image_name))
                self.image_labels.append(self.categories.index(category))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        # Add an additional dimension for time (T=1)
        image = image.unsqueeze(0)  # Now image shape is (1, 3, 224, 224)

        return image, self.image_labels[idx]

def get_imagenet_val_dataloader():
    val_dataset = ImageNetDataset('/home/katop1234/Datasets/ilsvrc/val/')
    num_val_samples = 1000  # The number of samples you want to evaluate
    indices = list(range(len(val_dataset)))
    np.random.seed(4)
    np.random.shuffle(indices)
    val_sampler = SubsetRandomSampler(indices[:num_val_samples])
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, num_workers=14)
    return val_loader

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
    # TODO make this compatible with where the automated scripts download all the files
    if ds_type == 'image':
        transforms_train = get_image_transforms()
        if name == 'cvf':
            dataset_train = datasets.ImageFolder(os.path.join(root_path, 'arxiv_resized_train_val_split/train/'),
                                                 transform=transforms_train)
        elif name == 'imagenet':
            dataset_train = datasets.ImageFolder('/shared/group/ilsvrc/train', 
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
            dataset_train = VideoDataset(path_to_data_dir="/shared/group/kinetics/")
        elif name == "Objectron":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/Objectron/")
        elif name == "SSV2":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/SSV2_videos/") 
        elif name == "UCF101":
            dataset_train = VideoDataset(path_to_data_dir="/shared/katop1234/Datasets/UCF101/") 
        elif name == "CSV":
            dataset_train = VideoDataset(path_to_data_dir="/shared/dannyt123/Datasets/CSV")
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Wrong dataset type.")

    return dataset_train

def combined_gen(image_itr_cls, video_itr_cls, accum_iter_img, accum_iter_vid, image_itr, video_itr, num_iter):
    i = 0
    if image_itr_cls:
        image_gen = iter(image_itr_cls)
    if video_itr_cls:
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
        if video_itr == 0:
            self.num_iter_per_epoch = 96*(accum_iter_img*image_itr)
        else:
            self.num_iter_per_epoch = 24*(accum_iter_img*image_itr + accum_iter_vid*video_itr)

    def __iter__(self):
        return combined_gen(self.image_gen, self.video_gen, self.accum_iter_img, self.accum_iter_vid, self.image_itr, self.video_itr, len(self))

    def __len__(self):
        return self.num_iter_per_epoch # carefuly computed to avoid accum_iter updates here


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, dataset_list, dataset_conf, ds_type):
        dataset_conf = [float(x) for x in dataset_conf]
        datasets = [get_dataset(ds_name, root_path, ds_type) for ds_name in dataset_list]
        conf = [i / sum(dataset_conf) for i in dataset_conf]
        self.datasets = datasets
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
