# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import PIL
import warnings
from pathlib import Path
from typing import Any, Callable, cast, Optional, Union

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets import DatasetFolder, VisionDataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


# Some files are corrupted in IN21K
CORRUPT_FILES = set()
if not os.path.isfile(file_path):
    sys.exit(f"Error: File '{file_path}' not found! Create one")
with open('corrupt_imagenet_files.txt', 'r') as f:
    for line in f:
        CORRUPT_FILES.add(line.strip())

def is_valid_image(path: str) -> bool:
    if path in CORRUPT_FILES:
        return False
    return True


class FilteredINVal(DatasetFolder):
    def __init__(
        self,
        root: Union[str, Path],
        class_to_idx_21K: dict,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> None:
        VisionDataset.__init__(self, root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)

        # ignore n04399382 (present in 1k-val but not in 21k)
        del class_to_idx["n04399382"]
        classes.remove("n04399382")

        # map 1k idx to 21k idx
        for k in class_to_idx:
            class_to_idx[k] = class_to_idx_21K[k]

        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples


def build_dataset(is_train, args, c2i21k=None):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == "IMNET21K":
        nb_classes = 19167
        if is_train:
            # use 21K
            root = "/mnt/data/Public_datasets/ImageNet21K/winter21_whole/"
            dataset = datasets.ImageFolder(root, transform=transform, is_valid_file=is_valid_image)
        else:
            # use 1k-val, accuracy metric is going to be wrong (not calculated on 999 classes, but still an indicator)
            root = "/mnt/data/Public_datasets/imagenet/imagenet_pytorch/val/"
            dataset = FilteredINVal(root, class_to_idx_21K=c2i21k, transform=transform)
            assert len(dataset.classes) == 999

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
