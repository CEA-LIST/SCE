# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from omegaconf import DictConfig
from typing import List, Optional

from sce.datamodule.folder import FolderDataModule


class ImagenetDataModule(FolderDataModule):
    """Base data module for the Imagenet dataset.

    Args:
        datadir (str): Where to load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.

    Example::

        datamodule = ImagenetDataModule(datadir)
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
        return 1000


class Imagenet100DataModule(FolderDataModule):
    """Base data module for the Imagenet100 dataset.

    Args:
        datadir (str): Where to load the data.
        train (Optional[DictConfig]): Configuration for the training data to define the loading of data, the transforms and the dataloader. Defaults to None.
        val (Optional[DictConfig]): Configuration for the validation data to define the loading of data, the transforms and the dataloader. Defaults to None.
        test (Optional[DictConfig]): Configuration for the testing data to define the loading of data, the transforms and the dataloader. Defaults to None.

    Example::

        datamodule = Imagenet100DataModule(datadir)
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)

    def _verify_classes(self, split: str = 'train') -> None:
        split_dir = self.datadir / split
        dirs = [dir.stem for dir in split_dir.iterdir() if dir.is_dir()]

        assert all([class_ in dirs for class_ in self.class_list]), f"error classes found: {dirs}"
    
    @property
    def class_list(self) -> List[str]:
        return [
            'n07836838', 'n04111531', 'n04493381', 'n02093428', 'n04067472', 'n01773797', 'n02108089', 'n02113978', 'n03930630', 'n02085620', 'n02138441', 'n02090622', 'n04238763', 'n03637318', 'n01692333', 'n02804414', 'n02113799', 'n01978455', 'n02089973', 'n01749939', 'n03794056', 'n03017168', 'n04435653', 'n03785016', 'n02114855', 'n04336792', 'n02259212', 'n03775546', 'n01558993', 'n03891251', 'n03777754', 'n02109047', 'n02326432', 'n02091831', 'n02123045', 'n03642806', 'n02119022', 'n01729322', 'n02105505', 'n04026417', 'n03494278', 'n03584829', 'n02231487', 'n03085013', 'n04229816', 'n07714571', 'n04429376', 'n03594734', 'n04517823', 'n01855672', 'n02018207', 'n03259280', 'n03837869', 'n03424325', 'n03764736', 'n04592741', 'n02104029', 'n04127249', 'n02100583', 'n13040303', 'n03062245', 'n02087046', 'n02869837', 'n04485082', 'n02172182', 'n02396427', 'n03787032', 'n03903868', 'n02107142', 'n02788148', 'n02974003', 'n02106550', 'n03492542', 'n03530642', 'n02086240', 'n02859443', 'n03379051', 'n01735189', 'n04589890', 'n07753275', 'n04136333', 'n02089867', 'n04099969', 'n03032252', 'n02483362', 'n03947888', 'n02488291', 'n04418357', 'n01983481', 'n01820546', 'n01980166', 'n02086910', 'n02701002', 'n02009229', 'n02877765', 'n07831146', 'n07715103', 'n13037406', 'n02116738', 'n02099849'
        ]

    @property
    def num_classes(self) -> int:
        return 100
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)
