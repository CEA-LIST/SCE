# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from abc import ABC, abstractproperty
import hydra
from omegaconf import DictConfig
from typing import Optional

import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset
from torchvision.transforms import ToTensor

from sce.datamodule.base import BaseDataModule
from sce.datasets.dict_dataset import DictDataset


class CIFARDataModule(BaseDataModule, ABC):
    """Base data module for the CIFAR datasets.

    Args:
        datadir (str): Where to save/load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.
        split_train_ratio (Optional[float]): If not None randomly split the train dataset in two with split_train_ration ratio for train. Defaults to None.
        seed_for_split (int): Seed for the split. Defaults to 42.
    
    Attributes:
        DATASET: should be defined by subclasses.
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        split_train_ratio: Optional[float] = None,
        seed_for_split: int = 42
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)
        self.split_train_ratio = split_train_ratio
        self.seed_for_split = seed_for_split

    @abstractproperty
    def DATASET(self) -> VisionDataset:
        return None

    def prepare_data(self) -> None:
        self.DATASET(self.datadir, train=True, download=True,
                     transform=ToTensor())
        self.DATASET(self.datadir, train=False, download=True,
                     transform=ToTensor())

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError(
                    'No training configuration has been passed.')

            self.train_transform = hydra.utils.instantiate(
                self.train.transform)
            rank_zero_info(f"Train transform: {self.train_transform}")

            if self.split_train_ratio is None:
                self.train_dataset = DictDataset(
                    self.DATASET(self.datadir, train=True,
                                 download=False, transform=self.train_transform)
                )

                if self.val is not None:
                    self.val_transform = hydra.utils.instantiate(
                        self.val.transform)
                    rank_zero_info(f"Val transform: {self.val_transform}")
                    self.val_dataset = DictDataset(
                        self.DATASET(self.datadir, train=False,
                                     download=False, transform=self.val_transform)
                    )
            else:
                train_dataset = self.DATASET(self.datadir, train=True,
                                 download=False, transform=self.train_transform)
                train_length = round(len(train_dataset) * self.split_train_ratio)
                val_length = len(train_dataset) - train_length
                train_dataset, val_dataset = random_split(train_dataset, [train_length, val_length], torch.Generator().manual_seed(self.seed_for_split))
                self.train_dataset = DictDataset(train_dataset)

                if self.val is not None:
                    self.val_transform = hydra.utils.instantiate(
                        self.val.transform)
                    rank_zero_info(f"Val transform: {self.val_transform}")
                    self.val_dataset = val_dataset
                    self.val_dataset.transform = self.val_transform
                    self.val_dataset = DictDataset(val_dataset)

        elif stage == "test":
            if self.test is None:
                raise RuntimeError('No testing configuration has been passed.')

            self.test_transform = hydra.utils.instantiate(self.test.transform)
            rank_zero_info(f"Test transform: {self.test_transform}")
            self.test_dataset = DictDataset(
                self.DATASET(self.datadir, train=False,
                             download=False, transform=self.test_transform)
            )


class CIFAR10DataModule(CIFARDataModule):
    """Data module for the CIFAR10 dataset.

    Args:
        datadir (str): Where to save/load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.

    Example::

        datamodule = CIFAR10DataModule(datadir)
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)

    @property
    def DATASET(self) -> VisionDataset:
        return CIFAR10

    @property
    def num_classes(self) -> int:
        return 10


class CIFAR100DataModule(CIFARDataModule):
    """Data module for the CIFAR100 dataset.

    Args:
        datadir (str): Where to save/load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.

    Example::

        datamodule = CIFAR100DataModule(datadir)
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)

    @property
    def DATASET(self) -> VisionDataset:
        return CIFAR100

    @property
    def num_classes(self) -> int:
        return 100
