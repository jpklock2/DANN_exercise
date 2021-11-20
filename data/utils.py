"""Utility functions for data."""
import os
import glob

import torch
import torchvision
import torchvision.transforms as transforms

from data.dataset import MNIST_M, Office, SynSigns, GTSRB

import torch.utils.data as data_utils
import random
import itertools
import numpy as np

def load_mnist(pre_load_files=0, get_subset=1, **kwargs):
    """Load MNIST dataloader, repeating along 3 channels.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)),
         # Make 3D images from grayscal MNIST
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)

    if (get_subset == 1):
        # Getting subset
        total_idx = []
        for i in range(0, 10):
          idx = list(np.where((trainset.targets == i))[0])
          random_idx = random.sample(idx, 100)
          total_idx.append(random_idx)
        total_idx = list(itertools.chain.from_iterable(total_idx))
        trainset = data_utils.Subset(trainset, total_idx)

    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)

    if (get_subset == 1):
        # Getting subset
        total_idx = []
        for i in range(0, 10):
          idx = list(np.where((testset.targets == i))[0])
          random_idx = random.sample(idx, 100)
          total_idx.append(random_idx)
        total_idx = list(itertools.chain.from_iterable(total_idx))
        testset = data_utils.Subset(testset, total_idx)

    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def load_svhn(pre_load_files=0, get_subset=1, **kwargs):
    """Load SVHM dataloader.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    transform = transforms.Compose(
        [transforms.Resize(28),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    trainset = torchvision.datasets.SVHN(root='./data',
                                         split='train',
                                         download=True,
                                         transform=transform)

    if (get_subset == 1):
        # Getting subset
        total_idx = []
        for i in range(0, 10):
          idx = list(np.where((trainset.labels == i))[0])
          random_idx = random.sample(idx, 100)
          total_idx.append(random_idx)
        total_idx = list(itertools.chain.from_iterable(total_idx))
        trainset = data_utils.Subset(trainset, total_idx)

    testset = torchvision.datasets.SVHN(root='./data',
                                        split='test',
                                        download=True,
                                        transform=transform)

    if (get_subset == 1):
        # Getting subset
        total_idx = []
        for i in range(0, 10):
          idx = list(np.where((testset.labels == i))[0])
          random_idx = random.sample(idx, 100)
          total_idx.append(random_idx)
        total_idx = list(itertools.chain.from_iterable(total_idx))
        testset = data_utils.Subset(testset, total_idx)

    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def load_synsigns(pre_load_files=0, get_subset=1, **kwargs):
    """Load SynSigns dataloader.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    fullset = SynSigns(pre_load_files,
                       'data/synsigns',
                       'train_labelling.txt',
                       transform=transform)

    # Split the dataset, 80% for training and 20% for testing
    trainset_length = int(len(fullset) * 0.8)
    testset_lenght = len(fullset) - trainset_length
    lengths = [trainset_length, testset_lenght]

    trainset, testset = torch.utils.data.random_split(fullset, lengths)

    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def load_gtsrb(pre_load_files=0, get_subset=1, **kwargs):
    """Load GTSRB dataloader.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    train_folders = glob.glob('data/GTSRB/Final_Training/Images/*')

    transform = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    fullset = torch.utils.data.ConcatDataset(
            [GTSRB(pre_load_files, folder, transform=transform) for folder in train_folders])

    # Split the dataset, 80% for training and 20% for testing
    trainset_length = int(len(fullset) * 0.8)
    testset_lenght = len(fullset) - trainset_length
    lengths = [trainset_length, testset_lenght]

    trainset, testset = torch.utils.data.random_split(fullset, lengths)

    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def load_office(pre_load_files=0, get_subset=1, dataset='amazon', **kwargs):
    """Load an Office Dataset.

    :dataset: TODO
    :**kwargs: TODO
    :returns: TODO
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop([227, 227]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop([227, 227]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = Office(pre_load_files,
                      data_root=os.path.join('data', 'office'),
                      dataset=dataset,
                      transform=train_transform)

    testset = Office(pre_load_files,
                     data_root=os.path.join('data', 'office'),
                     dataset=dataset,
                     transform=test_transform)

    # Use sane dataset for testing and training as contains few images
    return get_loader(trainset, **kwargs), get_loader(testset, **kwargs)


def load_mnist_m(pre_load_files=0, get_subset=1, **kwargs):
    """Load MNIST_M dataloader.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    img_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = MNIST_M(
        pre_load_files,
        data_root=os.path.join('data', 'mnist_m', 'mnist_m_train'),
        data_list=os.path.join('data', 'mnist_m', 'mnist_m_train_labels.txt'),
        transform=img_transform
    )

    testset = MNIST_M(
        pre_load_files,
        data_root=os.path.join('data', 'mnist_m', 'mnist_m_test'),
        data_list=os.path.join('data', 'mnist_m', 'mnist_m_test_labels.txt'),
        transform=img_transform
    )
    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def get_loader(dataset, **kwargs):
    """Get dataloader from dataset."""
    return torch.utils.data.DataLoader(dataset, **kwargs)