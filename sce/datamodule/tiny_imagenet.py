# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from omegaconf import DictConfig
from typing import Optional

from sce.datamodule.folder import FolderDataModule


class TinyImagenetDataModule(FolderDataModule):
    """Base data module for the Tiny Imagenet dataset.

    Args:
        datadir (str): Where to load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.
    
    Example::

        datamodule = TinyImagenetDataModule(datadir)
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
    def num_classes(self) -> int:
        return 200
