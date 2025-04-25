# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from torch import NoneType

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.distributed as dist

from PIL import Image,ImageFile
from torch.utils import data
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import torchvision.transforms as T
import random
import math
import numpy as np
import torch

# for worldview dataset
#from osgeo import gdal


class MillionAIDDataset_full(data.Dataset):
    def __init__(self, root, train=True, transform=None, tag=100):

        print(os.getcwd())

        with open(os.path.join(root, 'files.txt'.format(tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            #fname, _, idx = item.strip().split()
            fname = item.strip()#.split()
            trn_files.append(os.path.join(root + '/all_img', fname))
            trn_targets.append(0)

        #with open(os.path.join(root, 'val.txt'), mode='r') as f:
        #    valid_infos = f.readlines()
        #f.close()

        #val_files = []
        #val_targets = []

        #for item in valid_infos:
        #    fname, _, idx = item.strip().split()
        #    val_files.append(os.path.join(root + '/all_img', fname))
        #    val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        #else:
        #    self.files = val_files
        #    self.targets = val_targets
        
        self.transform = transform

        print('Creating MillionAID dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]
        
        img = Image.open(img_path)

        #if self.transform != None:

        img = self.transform(img)

        return img, self.targets[i]    

class MillionAIDDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, tag=100):

        print(os.getcwd())

        with open(os.path.join(root, 'train.txt'.format(tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, _, idx = item.strip().split()
            trn_files.append(os.path.join(root + '/all_img', fname))
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'val.txt'), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, _, idx = item.strip().split()
            val_files.append(os.path.join(root + '/all_img', fname))
            val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        else:
            self.files = val_files
            self.targets = val_targets

        self.transform = transform

        print('Creating MillionAID dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]

        img = Image.open(img_path)

        #if self.transform != None:

        img = self.transform(img)

        return img, self.targets[i]

class UCMDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, tag=None, trainfile=None, valfile=None):

        if trainfile is not None:
            with open(trainfile, mode='r') as f:
                train_infos = f.readlines()

        else:
            with open(os.path.join(root, 'train.txt'), mode='r') as f:
                train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, _, idx = item.strip().split()
            trn_files.append(os.path.join(root + '/all_img', fname))
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'val.txt'), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, _, idx = item.strip().split()
            val_files.append(os.path.join(root + '/all_img', fname))
            val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        else:
            self.files = val_files
            self.targets = val_targets

        self.transform = transform

        print('Creating UCM dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]

        img = Image.open(img_path)

        if self.transform != None:

            img = self.transform(img)

        return img, self.targets[i]

class AIDDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, tag=None):

        with open(os.path.join(root, 'train.txt'.format(split,tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, _, idx = item.strip().split()
            trn_files.append(os.path.join(root + '/all_img', fname))
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'val.txt'.format(split,tag)), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, _, idx = item.strip().split()
            val_files.append(os.path.join(root + '/all_img', fname))
            val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        else:
            self.files = val_files
            self.targets = val_targets

        self.transform = transform

        print('Creating AID dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]

        img = Image.open(img_path)

        if self.transform != None:

            img = self.transform(img)

        return img, self.targets[i]

class NWPURESISCDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, tag=None):

        with open(os.path.join(root, 'train.txt'.format(split,tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, _, idx = item.strip().split()
            trn_files.append(os.path.join(root + '/all_img', fname))
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'val.txt'.format(split,tag)), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, _, idx = item.strip().split()
            val_files.append(os.path.join(root + '/all_img', fname))
            val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        else:
            self.files = val_files
            self.targets = val_targets

        self.transform = transform

        print('Creating NWPU_RESISC45 dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]

        img = Image.open(img_path)

        if self.transform != None:

            img = self.transform(img)

        return img, self.targets[i]

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset == 'millionaid':
        print('Loading MillionAID dataset!')
        args.nb_classes = 51
        dataset = MillionAIDDataset(args.data_path, train=is_train, transform=transform, tag=args.tag)
    elif args.dataset == 'ucm':
        print('Loading UCM dataset!')
        args.nb_classes = 21
        dataset = UCMDataset(args.data_path, train=is_train, transform=transform, tag=args.tag)
    elif args.dataset == 'aid':
        print('Loading AID dataset!')
        args.nb_classes = 30
        dataset = AIDDataset(args.data_path, train=is_train, transform=transform, split=args.split, tag=args.tag)
    elif args.dataset == 'nwpu':
        print('Loading NWPU-RESISC45 dataset!')
        args.nb_classes = 45
        dataset = NWPURESISCDataset(args.data_path, train=is_train, transform=transform, split=args.split, tag=args.tag)
    else:
        raise NotImplementedError

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    # if args.input_size <= 224:
    crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
