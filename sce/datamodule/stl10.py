# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1


import hydra
from omegaconf import DictConfig
from typing import Optional

from torchvision import transforms
from torchvision.datasets import STL10

from sce.datamodule.base import BaseDataModule
from sce.datasets.dict_dataset import DictDataset


class STL10DataModule(BaseDataModule):
    """Data module for the STL10 dataset in SSL setting.

    Args:
        datadir (str): Where to save/load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.
        folds (Optional[int]): One of {0-9} or None. For training, loads one of the 10 pre-defined folds of 1k samples for the standard         evaluation procedure. If no value is passed, loads the 5k samples. Defaults to None.
        training_split (str): Split used for the training dataset. Defaults to train+unlabeled.

    Example::

        datamodule = STL10DataModule(datadir)
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        folds: Optional[int] = None,
        training_split: str = 'unlabeled'
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)
        self.folds = folds
        self.training_split = training_split

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """
        Downloads the unlabeled, train and test split
        """
        STL10(self.datadir, split='unlabeled', download=True,
              transform=transforms.ToTensor())
        STL10(self.datadir, folds=self.folds, split='train', download=True,
              transform=transforms.ToTensor())
        STL10(self.datadir, split='test', download=True,
              transform=transforms.ToTensor())

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            assert self.training_split in [
                'train', 'train+unlabeled', 'unlabeled']

            self.train_transform = hydra.utils.instantiate(
                self.train.transform)
            self.train_dataset = DictDataset(
                STL10(self.datadir, folds=self.folds, split=self.training_split,
                      download=False, transform=self.train_transform)
            )

            if self.val is not None:
                self.val_transform = hydra.utils.instantiate(
                    self.val.transform)
                self.val_dataset = DictDataset(
                    STL10(self.datadir, split='test', download=False, transform=self.val_transform)
                )

        elif stage == "test":
            self.test_transform = hydra.utils.instantiate(
                self.test.transform)
            self.test_dataset = DictDataset(
                STL10(self.datadir, split='test', transform=self.test_transform)
            )
