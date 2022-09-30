# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torch.utils.data.dataloader import DataLoader

from sce.utils.strategies import get_local_batch_size_in_trainer



class BaseDataModule(LightningDataModule, ABC):
    """Abstract class that inherits from LightningDataModule to follow standardized preprocessing for all datamodules in sce.

    Args:
        datadir (str): Where to save/load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.

    Example::

        import hydra
        from omegaconf import OmegaConf

        class MyDataModule(BaseDataModule):
            def __init__(self, datadir, train=None, val=None, test=None):
                super().__init__(datadir, train, val, test)

            def train_dataloader(self):
                transform = hydra.utils.instantiate(self.train.transform)


        train = OmegaConf.create({
            'transform' : {'_target_': 'sce.transforms.ToTensor()'}, # transform configuration
            'loader': {'drop_last': True, ...} # dataloader configuration
            'global_batch_size': 256
            }
        )

        my_data_module = MyDataModule('data/', train=train)
        train_dataloader = my_data_module.train_dataloader()

    .. warning::
            The loader subconfigurations must not contain 'batch_size' that is automatically computed from the 'global_batch_size' specified in the configuration.
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.datadir = Path(datadir)

        train = self._validate_train_config(train)
        val = self._validate_val_config(val)
        test = self._validate_test_config(test)

        self.train = train
        self.val = val
        self.test = test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # If no train configuration passed, override train_dataloader method by LightningDataModule's to prevent the trainer from detecting a training dataloader.
        if train is None:
            self.train_dataloader = super().train_dataloader
        #         
        # If no val configuration passed, override val_dataloader method by LightningDataModule's to prevent the trainer from detecting a validation dataloader.
        if val is None:
            self.val_dataloader = super().val_dataloader

        # If no test configuration passed, override val_dataloader method by LightningDataModule's to prevent the trainer from detecting a test dataloader.
        if test is None:
            self.test_dataloader = super().test_dataloader
    
    def _validate_train_config(self, cfg: Optional[DictConfig]):
        if cfg is None:
            return cfg

        if cfg.get("loader"):
            if cfg.loader.get("collate_fn"):
                raise NotImplementedError('No custom collate_fn supported')
            else:
                self.train_collate_fn = None
            if cfg.loader.get("batch_size"):
                cfg.loader.pop("batch_size")
                rank_zero_warn("Batch size has been remove for train loader config because global_batch_size to train config should be passed.")
        return cfg
    
    def _validate_val_config(self, cfg: DictConfig):
        if cfg is None:
            return cfg

        if cfg.get("loader"):
            if cfg.loader.get("collate_fn"):
                raise NotImplementedError('No custom collate_fn supported')
                cfg.loader.pop("collate_fn")
            else:
                self.val_collate_fn = None
            if cfg.loader.get("batch_size"):
                cfg.loader.pop("batch_size")
                rank_zero_warn("Batch size has been remove for val loader config because global_batch_size to val config should be passed.")
        return cfg
    
    def _validate_test_config(self, cfg: DictConfig):
        if cfg is None:
            return cfg

        if cfg.get("loader"):
            if cfg.loader.get("collate_fn"):
                raise NotImplementedError('No custom collate_fn supported')
                cfg.loader.pop("collate_fn")
            else:
                self.test_collate_fn = None
            if cfg.loader.get("batch_size"):
                cfg.loader.pop("batch_size")
                rank_zero_warn("Batch size has been remove for test loader config because global_batch_size to test config should be passed.")
        return cfg

    @property
    @abstractmethod
    def num_classes(self) -> int:
        return -1

    @property
    def train_num_samples(self) -> int:
        return len(self.train_dataset) if self.train_dataset else 0

    @property
    def val_num_samples(self) -> int:
        return len(self.val_dataset) if self.val_dataset else 0

    @property
    def test_num_samples(self) -> int:
        return len(self.test_dataset) if self.test_dataset else 0

    @property
    def train_global_batch_size(self) -> int:
        if not self.train.get('global_batch_size'):
            raise AttributeError(
                'global_batch_size should be defined in train datamodule config.')
        return self.train.global_batch_size

    @property
    def val_global_batch_size(self) -> int:
        if not self.val.get('global_batch_size'):
            raise AttributeError(
                'global_batch_size should be defined in val datamodule config.')
        return self.val.global_batch_size

    @property
    def test_global_batch_size(self) -> int:
        if not self.test.get('global_batch_size'):
            raise AttributeError(
                'global_batch_size should be defined in test datamodule config.')
        return self.test.global_batch_size

    @property
    def train_local_batch_size(self) -> int:
        if self.trainer is not None:
            return get_local_batch_size_in_trainer(self.train_global_batch_size, self.trainer)
        else:
            return self.train_global_batch_size

    @property
    def val_local_batch_size(self) -> int:
        if self.trainer is not None:
            return get_local_batch_size_in_trainer(self.val_global_batch_size, self.trainer)
        else:
            return self.val_global_batch_size

    @property
    def test_local_batch_size(self) -> int:
        if self.trainer is not None:
            return get_local_batch_size_in_trainer(self.test_global_batch_size, self.trainer)
        else:
            return self.test_global_batch_size

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            return RuntimeError('No passed training configuration so dataloader cannot be retrieved.')

        loader = DataLoader(
            self.train_dataset, batch_size=self.train_local_batch_size, collate_fn=self.train_collate_fn, **self.train.loader)
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val is None:
            return RuntimeError('No passed validation configuration so dataloader cannot be retrieved.')
        
        loader = DataLoader(
            self.val_dataset, batch_size=self.val_local_batch_size, collate_fn=self.val_collate_fn, **self.val.loader)
        return loader

    def test_dataloader(self) -> DataLoader:
        if self.test is None:
            return RuntimeError('No passed testing configuration so dataloader cannot be retrieved.')

        loader = DataLoader(
            self.test_dataset, batch_size=self.test_local_batch_size, collate_fn=self.test_collate_fn, **self.test.loader)
        return loader
