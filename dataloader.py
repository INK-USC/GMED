import torch
import yaml
import numpy as np
import pickle
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from tqdm import tqdm
from yacs.config import CfgNode
from maskrcnn_benchmark.config import cfg as maskrcnn_cfg
from utils.tupperware import tupperware
from utils.build_transforms import build_transforms
from data.benchmark_mir import CLDataLoader, get_permuted_mnist, get_split_mnist, get_miniimagenet, get_rotated_mnist, \
    get_split_cifar10, get_split_cifar100, IIDDataset, FuzzyCLDataLoader

from utils.utils import DotDict, get_config_attr

import random

_dataset = {

}

_smnist_loaders = None
def get_split_mnist_dataloader(cfg, split='train', filter_obj=None, batch_size=128, *args, **kwargs):
    fuzzy = get_config_attr(cfg,'EXTERNAL.OCL.FUZZY', default=0, mute=True)
    d = DotDict()
    global _smnist_loaders
    if not _smnist_loaders:
        data = get_split_mnist(d, cfg)
        loader_cls = CLDataLoader if not fuzzy else FuzzyCLDataLoader
        train_loader, val_loader, test_loader = [loader_cls(elem, batch_size, train=t) \
            for elem, t in zip(data, [True, False, False])]
        _smnist_loaders = train_loader, val_loader, test_loader
    else:
        train_loader, val_loader, test_loader = _smnist_loaders

    if split == 'train':
        return train_loader[filter_obj[0]]
    elif split == 'val':
        return val_loader[filter_obj[0]]
    elif split == 'test':
        return test_loader[filter_obj[0]]


_rmnist_loaders = None
def get_rotated_mnist_dataloader(cfg, split='train', filter_obj=None, batch_size=128, task_num=10, *args, **kwargs):
    d = DotDict()
    fuzzy = get_config_attr(cfg, 'EXTERNAL.OCL.FUZZY', default=0, mute=True)
    global _rmnist_loaders
    if not _rmnist_loaders:
        data = get_rotated_mnist(d)
        #train_loader, val_loader, test_loader = [CLDataLoader(elem, batch_size, train=t) \
        #                                         for elem, t in zip(data, [True, False, False])]
        loader_cls = CLDataLoader if not fuzzy else FuzzyCLDataLoader
        train_loader, val_loader, test_loader = [loader_cls(elem, batch_size, train=t) \
                                                 for elem, t in zip(data, [True, False, False])]
        _rmnist_loaders = train_loader, val_loader, test_loader
    else:
        train_loader, val_loader, test_loader = _rmnist_loaders
    if split == 'train':
        return train_loader[filter_obj[0]]
    elif split == 'val':
        return val_loader[filter_obj[0]]
    elif split == 'test':
        return test_loader[filter_obj[0]]

_pmnist_loaders = None
def get_permute_mnist_dataloader(cfg, split='train', filter_obj=None, batch_size=128, task_num=10, *args, **kwargs):
    d = DotDict()
    fuzzy = get_config_attr(cfg, 'EXTERNAL.OCL.FUZZY', default=0, mute=True)
    global _pmnist_loaders
    if not _pmnist_loaders:
        data = get_permuted_mnist(d)
        loader_cls = CLDataLoader if not fuzzy else FuzzyCLDataLoader
        train_loader, val_loader, test_loader = [loader_cls(elem, batch_size, train=t) \
            for elem, t in zip(data, [True, False, False])]
        _pmnist_loaders = train_loader, val_loader, test_loader
    else:
        train_loader, val_loader, test_loader = _pmnist_loaders
    if split == 'train':
        return train_loader[filter_obj[0]]
    elif split == 'val':
        return val_loader[filter_obj[0]]
    elif split == 'test':
        return test_loader[filter_obj[0]]

_cache_cifar = None
def get_split_cifar_dataloader(cfg, split='train', filter_obj=None, batch_size=128, *args, **kwargs):
    d = DotDict()
    fuzzy = get_config_attr(cfg, 'EXTERNAL.OCL.FUZZY', default=0, mute=True)
    global _cache_cifar
    if not _cache_cifar:
        data = get_split_cifar10(d,cfg) #ds_cifar10and100(batch_size=batch_size, num_workers=0, cfg=cfg, **kwargs)
        loader_cls = CLDataLoader if not fuzzy else FuzzyCLDataLoader
        train_loader, val_loader, test_loader = [loader_cls(elem, batch_size, train=t) \
                                                 for elem, t in zip(data, [True, False, False])]
        _cache_cifar = train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = _cache_cifar
    if split == 'train':
        return train_loader[filter_obj[0]]
    elif split == 'val':
        return val_loader[filter_obj[0]]
    elif split == 'test':
        return test_loader[filter_obj[0]]

_cache_cifar100 = None
def get_split_cifar100_dataloader(cfg, split='train', filter_obj=None, batch_size=128, *args, **kwargs):
    d = DotDict()
    fuzzy = get_config_attr(cfg, 'EXTERNAL.OCL.FUZZY', default=0, mute=True)
    global _cache_cifar100
    if not _cache_cifar100:
        data = get_split_cifar100(d,cfg) #ds_cifar10and100(batch_size=batch_size, num_workers=0, cfg=cfg, **kwargs)
        loader_cls = CLDataLoader if not fuzzy else FuzzyCLDataLoader
        train_loader, val_loader, test_loader = [loader_cls(elem, batch_size, train=t) \
                                                 for elem, t in zip(data, [True, False, False])]
        _cache_cifar100 = train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = _cache_cifar100
    if split == 'train':
        return train_loader[filter_obj[0]]
    elif split == 'val':
        return val_loader[filter_obj[0]]
    elif split == 'test':
        return test_loader[filter_obj[0]]

_cache_mini_imagenet = None
def get_split_mini_imagenet_dataloader(cfg, split='train', filter_obj=None, batch_size=128, *args, **kwargs):
    global _cache_mini_imagenet
    d = DotDict()
    fuzzy = get_config_attr(cfg, 'EXTERNAL.OCL.FUZZY', default=0, mute=True)
    if not _cache_mini_imagenet:
        data = get_miniimagenet(d)
        loader_cls = CLDataLoader if not fuzzy else FuzzyCLDataLoader
        train_loader, val_loader, test_loader = [loader_cls(elem, batch_size, train=t) \
                                                 for elem, t in zip(data, [True, False, False])]
        _cache_mini_imagenet = train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = _cache_mini_imagenet
    if split == 'train':
        return train_loader[filter_obj[0]]
    elif split == 'val':
        return val_loader[filter_obj[0]]
    elif split == 'test':
        return test_loader[filter_obj[0]]

