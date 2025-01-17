import logging

from typing import List
from dataclasses import dataclass

import torch

from tqdm import tqdm

import oodd.utils

from oodd.datasets import DataModule
from .base_module import BaseModule


LOGGER = logging.getLogger(name=__file__)


@dataclass
class Checkpoint:
    """Class that holds attributes of a checkpoint.
    
    Instantiate with `path` and `populate()` the remaining fields and use `load()` to load model
    """
    path: str
    name: str = ""
    model: BaseModule = None
    optimizer: torch.optim.Optimizer = None
    datamodule: DataModule = None
    # others: dict = {
    #     "epoch": 0,
    #     "optimizer": dict(),
    #     "random": tuple(),
    #     "np_random": dict(),
    #     "torch": None,  # Tensor
    #     "torch_random": None,  # Tensor
    #     "cuda_random": None,  # Tensor
    #     "cuda_random_all": None,  # Tensor
    # }

    def load(self, device=oodd.utils.get_device()):
        self.load_model(device=device)
        self.load_datamodule(distributed=False)
        return self

    def load_DDP(self, rank: int, others_path: str):
        self.load_DDP_model(rank=rank)
        self.load_datamodule()
        self.load_others(others_path)
        return self

    def load_DDP_model(self, rank: int):
        self.model = oodd.models.load_DDP_model(self.path, rank=rank)
        return self

    def load_model(self, device=oodd.utils.get_device()):
        self.model = oodd.models.load_model(self.path, device=device)
        return self

    def load_datamodule(self, **override_kwargs):
        LOGGER.info('Loading DataModule' + f' with overridding kwargs {override_kwargs}' if override_kwargs else '')
        self.datamodule = DataModule.load(self.path, **override_kwargs)
        return self

    def load_others(self, path):
        LOGGER.info("Loading other info")
        self.others = torch.load(path)
        return self


@dataclass
class CheckpointList:
    """Class that holds a list of Checkpoints"""
    checkpoints: List[Checkpoint]

    def __getitem__(self, idx):
        return self.checkpoints[idx]

    def get_from_uuid(self, path):
        checkpoint = [checkpoint for checkpoint in self.checkpoints if checkpoint.path == path]
        assert len(checkpoint) <= 1, "Several checkpoints with this UUID!"
        if len(checkpoint) == 1:
            return checkpoint[0]
        return None

    def load(self, device=oodd.utils.get_device()):
        for checkpoint in tqdm(self.checkpoints):
            checkpoint.load(device=device)

    def load_model(self, device=oodd.utils.get_device()):
        for checkpoint in tqdm(self.checkpoints):
            checkpoint.load_model(device=device)

    def load_datamodule(self, **override_kwargs):
        for checkpoint in tqdm(self.checkpoints):
            checkpoint.load_datamodule(**override_kwargs)

    def __len__(self):
        return len(self.checkpoints)
