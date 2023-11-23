"""Module with datasets from torchvision"""

import argparse
import logging
import os
from typing import Tuple, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision

import oodd.constants

from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, DATA_DIRECTORY
from oodd.datasets import transforms

from .dataset_base import BaseDataset


LOGGER = logging.getLogger(name=__file__)


TRANSFORM_NORMALIZE = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        transforms.Scale(a=0, b=1),  # Scale to [0, 1]
    ]
)


class EmbeddingDataset(BaseDataset):
    _train_embedding_filename = ""
    _test_embedding_filename = ""
    _train_label_filename = ""
    _test_label_filename = ""
    _split_args = dict()
    root_subdir = ""
    root_embdir = ""
    default_transform = lambda x: x

    def __init__(
        self,
        split=oodd.constants.TRAIN_SPLIT,
        root=DATA_DIRECTORY,
        transform=None,
        target_transform=None,
    ):
        super().__init__()

        self.split = split
        self.root = os.path.join(root, self.root_subdir, self.root_embdir)
        self.transform = self.default_transform if transform is None else transform
        self.target_transform = target_transform

        if self._split_args[split]["train"]:
            embedding_filename = self._train_embedding_filename
            label_filename = self._train_label_filename
        else:
            embedding_filename = self._test_embedding_filename
            label_filename = self._test_label_filename

        embedding_filepath = os.path.join(self.root, embedding_filename)
        label_filepath = os.path.join(self.root, label_filename)

        inps = np.load(embedding_filepath, allow_pickle=True)

        _shape = inps.shape
        if isinstance(_shape, tuple) and len(_shape) == 2:
            _, ed = _shape
            if ed == 512:
                inps = np.concatenate([inps, inps], axis=1).reshape(-1, 32, 32)
                inps = np.stack([inps, inps, inps], axis=3)  # (d, 32, 32, 3)
        inps = torch.from_numpy(inps)
        tgts = torch.from_numpy(np.load(label_filepath, allow_pickle=True))
        self.dataset = TensorDataset(inps, tgts)

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument("--root", type=str, default=DATA_DIRECTORY, help="Data storage location")
        return parser

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.numpy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)


class CIFAR10EmbeddingSimCLR(EmbeddingDataset):
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}, TEST_SPLIT: {"train": False}}
    _train_embedding_filename = "features_seed1.npy"
    _test_embedding_filename = "test_features_seed1.npy"
    _train_label_filename = "train_latents_targets.npy"
    _test_label_filename = "test_latents_targets.npy"
    root_subdir = "CIFAR10"
    root_embdir = "simclr"
    default_transform = TRANSFORM_NORMALIZE
