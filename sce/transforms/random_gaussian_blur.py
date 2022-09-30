# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import random
from PIL import ImageFilter, Image
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torchvision.transforms import GaussianBlur, RandomApply

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import BorderType
from kornia.filters import gaussian_blur2d


class RandomGaussianBlur(RandomApply):
    """Randomly apply gaussian blur to a tensor image with a certain probability.

    Args:
        kernel_size (Union[int, Iterable[int]]): Size of the gaussian kernel. Defaults to 23.
        sigma (Iterable[float]): radius for the gaussian kernel. Defaults to [0.1, 2.]. 
        p (float): Probability to apply color jitter. Defaults to 0.5.

    Example::

        transform = RandomGaussianBlur()
    """

    def __init__(
        self,
        kernel_size: Union[int, Iterable[int]] = 23,
        sigma: Iterable[float] = [.1, 2.],
        p: float = 0.5
    ) -> None:
        self.sigma = sigma
        self.p = p

        gaussian_blur = GaussianBlur(kernel_size, sigma=self.sigma)

        super().__init__(transforms=[gaussian_blur], p=self.p)


class RandomPILGaussianBlur(RandomApply):
    """Randomly apply PIL gaussian blur to a tensor image with a certain probability.

    Args:
        sigma (Iterable[float]): radius for the gaussian kernel. Defaults to [0.1, 2.]. 
        p (float): Probability to apply color jitter. Defaults to 0.5.

    Example::

        transform = RandomPILGaussianBlur()
    """

    def __init__(
        self,
        sigma: Iterable[float] = [.1, 2.],
        p: float = 0.5
    ) -> None:
        self.sigma = sigma
        self.p = p

        gaussian_blur = PILGaussianBlur(sigma=self.sigma)

        super().__init__(transforms=[gaussian_blur], p=p)


class PILGaussianBlur(object):
    """Gaussian blur augmentation in pillow."""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x: Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self) -> str:
        return f'{__class__.__name__}(sigma={self.sigma})'


class GPURandomGaussianBlurPil(IntensityAugmentationBase2D):
    r"""Apply gaussian blur given tensor image or a batch of tensor images randomly.

    .. image:: _static/img/RandomGaussianBlur.png

    Args:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``constant``, ``reflect``, ``replicate`` or ``circular``.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.filters.gaussian_blur2d`.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = "reflect",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self.flags = dict(kernel_size=kernel_size, sigma=sigma, border_type=BorderType.get(border_type))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ):
        sigma = torch.empty(1).uniform_(*self.flags["sigma"]).item()
        return gaussian_blur2d(
            input, self.flags["kernel_size"], (sigma, sigma), self.flags["border_type"].name.lower()
        )
