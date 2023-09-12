import os
from torch.utils.data import Dataset
from util.decoder import constants
from util.kinetics import Kinetics
from torchvision import datasets
from torchvision.transforms import transforms
from iopath.common.file_io import g_pathmgr as pathmgr
import numpy as np
import torch
import time
from pathlib import Path

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    dataset_csv=None, 
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    print('in new make_dataset')
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    if dataset_csv and dataset_csv.endswith('csv'):
        print('in csv')
        instances = []
        with pathmgr.open(dataset_csv, "r") as f:
            for path_label in enumerate(f.read().splitlines()):
                _, path = path_label
                if is_valid_file(path):
                    item = path, 0
                    instances.append(item)
    else: 
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

    return instances

class ImageFoldercsv(datasets.ImageFolder):
    def __init__(self, root, transform=None, dataset_csv=None):
        self.dataset_csv = dataset_csv
        
        super().__init__(root=root, transform=transform)
    
    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, dataset_csv=self.dataset_csv)

class VideoDataset(Kinetics):
    def __init__(self, path_to_data_dir, fb=False):
        super().__init__(path_to_data_dir=path_to_data_dir,
                         mode="pretrain",
                         sampling_rate=4,
                         num_frames=16,
                         train_jitter_scales=(256, 320),
                         repeat_aug=1,
                         jitter_aspect_relative=[0.75, 1.3333],
                         jitter_scales_relative=[0.5, 1.0],
                         fb=fb,)

def get_image_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=constants.mean, std=constants.std)])

def get_dataset(name, root_path, ds_type, fb=False):
    parent = Path(__file__).parent.absolute()
    datasets = os.path.join(parent, "datasets")
    start_time = time.time()
    if ds_type == 'image':
        transforms_train = get_image_transforms()
        if name == 'cvf':
<<<<<<< Updated upstream
            if fb:
                cvf_csv_path = os.path.join(datasets, 'cvf_images_fb.csv')
                dataset_train = datasets.ImageFolder('/private/home/amirbar/datasets/CVF/arxiv_resized_train_val_split/train/', 
                                                     transform=transforms_train, dataset_csv=cvf_csv_path)
            else:
                cvf_csv_path = os.path.join(datasets, 'cvf_images.csv')
                dataset_train = ImageFoldercsv('/shared/amir/dataset/arxiv_resized_train_val_split/train', 
                                                     transform=transforms_train, dataset_csv=cvf_csv_path)
        elif name == 'imagenet':
            if fb:
                imagenet_csv_path = os.path.join(datasets, 'imagenet_images_fb.csv')
                dataset_train = datasets.ImageFolder('/datasets01/imagenet_full_size/061417/train', 
                                                     transform=transforms_train, dataset_csv=imagenet_csv_path )
            else:
                imagenet_csv_path = os.path.join(datasets, 'imagenet_images.csv')
                dataset_train = ImageFoldercsv('/shared/group/ilsvrc/train', 
                                                    transform=transforms_train, dataset_csv=imagenet_csv_path)
=======
            # dataset_train = datasets.ImageFolder(os.path.join(root_path, 'arxiv_resized_train_val_split/train/'),
            #                                      transform=transforms_train)
            dataset_train = datasets.ImageFolder('/private/home/amirbar/datasets/CVF/arxiv_resized_train_val_split/train/', 
                                                 transform=transforms_train)
        elif name == 'imagenet':
            dataset_train = datasets.ImageFolder('/datasets01/imagenet_full_size/061417/train', 
                                                 transform=transforms_train)
>>>>>>> Stashed changes
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
                fb=fb,
            )
        elif name == "CrossTask":
<<<<<<< Updated upstream
            if fb:
                crosstask_csv_path = os.path.join(datasets, 'crosstask_videos_fb.csv')
            else:
                crosstask_csv_path = os.path.join(datasets, 'crosstask_videos.csv')
            dataset_train = VideoDataset(path_to_data_dir=crosstask_csv_path, fb=fb)
        elif name == "kinetics":
            if fb:
                kinetics_csv_path = os.path.join(datasets, 'kinetics_videos_fb.csv')
            else:
                kinetics_csv_path = os.path.join(datasets, 'kinetics_videos.csv')
            dataset_train = VideoDataset(path_to_data_dir=kinetics_csv_path, fb=fb)
=======
            dataset_train = VideoDataset(path_to_data_dir="/datasets01/CrossTask/053122/raw_video/")
        elif name == "kinetics":
            dataset_train = VideoDataset(path_to_data_dir="/datasets01/kinetics/092121/400/train_288px/")
>>>>>>> Stashed changes
        elif name == "Objectron":
            if fb:
                raise ValueError("Objectron not supported on fb")
            else:
                objectron_csv_path = os.path.join(datasets, 'objectron_videos.csv')
                dataset_train = VideoDataset(path_to_data_dir=objectron_csv_path, fb=fb)
        elif name == "SSV2":
<<<<<<< Updated upstream
            if fb:
                ssv2_csv_path = os.path.join(datasets, 'ssv2_videos_fb.csv')
            else:
                ssv2_csv_path = os.path.join(datasets, 'ssv2_videos.csv')
            dataset_train = VideoDataset(path_to_data_dir=ssv2_csv_path, fb=fb) 
        elif name == "UCF101":
            if fb:
                ucf101_csv_path = os.path.join(datasets, 'ucf101_videos_fb.csv')
            else:
                ucf101_csv_path = os.path.join(datasets, 'ucf101_videos.csv')
            dataset_train = VideoDataset(path_to_data_dir=ucf101_csv_path, fb=fb) 
=======
            dataset_train = VideoDataset(path_to_data_dir="/datasets01/SSV2/videos/") 
        elif name == "UCF101":
            dataset_train = VideoDataset(path_to_data_dir="/datasets01/ucf101/112018/data") 
>>>>>>> Stashed changes
        elif name == "CSV":
            if fb:
                csv_csv_path = os.path.join(datasets, 'csv_videos_fb.csv')
            else:
                csv_csv_path = os.path.join(datasets, 'csv_videos.csv')
            dataset_train = VideoDataset(path_to_data_dir=csv_csv_path, fb=fb)
        elif name == "EgoObj":
            if fb:
                egoobj_csv_path = os.path.join(datasets, 'egoobj_videos_fb.csv')
                dataset_train = VideoDataset(path_to_data_dir=egoobj_csv_path, fb=fb) 
            else:
                raise ValueError("Ego Object not implemented on BAIR")
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Wrong dataset type.")

    print('time for {name}: '.format(name=name), time.time() - start_time)
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
    def __init__(self, root_path, dataset_list, dataset_conf, ds_type, fb=False):
        dataset_conf = [float(x) for x in dataset_conf[0].split(',')]
        datasets = [get_dataset(ds_name, root_path, ds_type, fb) for ds_name in dataset_list[0].split(',')]
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
