# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import torchvision

from sce.datamodule.base import BaseDataModule
from sce.datamodule.cifar import CIFARDataModule, CIFAR10DataModule, CIFAR100DataModule
from sce.datamodule.folder import FolderDataModule
from sce.datamodule.imagenet import ImagenetDataModule, Imagenet100DataModule
from sce.datamodule.tiny_imagenet import TinyImagenetDataModule
from sce.datamodule.stl10 import STL10DataModule
