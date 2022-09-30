# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Optional

from torchvision.transforms.transforms import CenterCrop, Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

from sce.transforms.random_resized_crop import RandomResizedCrop


class LinearClassifierTrainTransform(Compose):
    """Define a transform for training a linear classifer evaluation protocol for self-supervised learning.

    Args:
        random_resized_crop (RandomResizedCrop): Random resized crop transform
        random_horizontal_flip (RandomHorizontalFlip): Random horizontal flip transform
        to_tensor (ToTensor): To tensor transform
        Normalize (Normalize): Normalize transform

    Example::

        random_resized_crop = RandomResizedCrop(...)
        random_horizontal_flip = RandomHorizontalFlip(...)
        to_tensor = ToTensor(...)
        normalize = Normalize(...)

        transform = LinearClassifierTrainTransform(
            random_resized_crop, random_horizontal_flip, to_tensor, normalize
        )
    """

    def __init__(
        self,
        random_resized_crop: RandomResizedCrop,
        random_horizontal_flip: RandomHorizontalFlip,
        to_tensor: ToTensor,
        normalize: Normalize,
    ) -> None:
        transforms = [random_resized_crop,
                      random_horizontal_flip, to_tensor, normalize]

        super().__init__(transforms=transforms)

class LinearClassifierSmallTrainTransform(Compose):
    """Define a transform for training a linear classifer evaluation protocol for self-supervised learning with small images datasets.

    Args:
        random_crop (RandomCrop): Random crop transform
        random_horizontal_flip (RandomHorizontalFlip): Random horizontal flip transform
        to_tensor (ToTensor): To tensor transform
        Normalize (Normalize): Normalize transform

    Example::

        random_crop = random_crop(...)
        random_horizontal_flip = RandomHorizontalFlip(...)
        to_tensor = ToTensor(...)
        normalize = Normalize(...)

        transform = LinearClassifierSmallTrainTransform(
            random_crop, random_horizontal_flip, to_tensor, normalize
        )
    """

    def __init__(
        self,
        random_crop: RandomCrop,
        random_horizontal_flip: RandomHorizontalFlip,
        to_tensor: ToTensor,
        normalize: Normalize,
    ) -> None:
        transforms = [random_crop,
                      random_horizontal_flip, to_tensor, normalize]

        super().__init__(transforms=transforms)


class LinearClassifierValTransform(Compose):
    """Define a transform for training a linear classifer evaluation protocol for self-supervised learning.

    Args:
        center_crop (CenterCrop): Center crop transform.
        to_tensor (ToTensor): To tensor transform.
        normalize (Normalize): Normalize transform.
        resize (Optional[Resize]): if not None, resize transform. Defaults to None.

    Example::

        resize = Resize(...) 
        center_crop = CenterCrop(...)
        to_tensor = ToTensor(...)
        normalize = Normalize(...)

        transform = LinearClassifierValTransform(
            resize, center_crop, to_tensor, normalize
        )
    """

    def __init__(
        self,
        center_crop: CenterCrop,
        to_tensor: ToTensor,
        normalize: Normalize,
        resize: Optional[Resize] = None,
    ) -> None:
        if resize == None:
            transforms = [center_crop, to_tensor, normalize]

        else:
            transforms = [resize, center_crop, to_tensor, normalize]

        super().__init__(transforms=transforms)
