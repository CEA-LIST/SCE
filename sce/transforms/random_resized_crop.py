# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Iterable, Union

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from sce.transforms.utils import _INTERPOLATION


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop the given image to random size and aspect ratio. If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size. This is popularly used to train the Inception networks.

    Args:
        size (Union[int, Iterable[int]]): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size (size, size) is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            In torchscript mode size as single int is not supported, use a sequence of length 1: [size, ].
        scale (Iterable[float]): scale range of the cropped image before resizing, relatively to the origin image. Defaults to [0.08, 1.].
        ratio (Iterable[float]): aspect ratio range of the cropped image before resizing. Defaults to [3/4, 4/3].
    interpolation (Union[str, InterpolationMode]): Desired interpolation enum defined by
        torchvision.transforms.InterpolationMode. If str type, lookup in _Interpolation dictionnary to retrieve the InterpolationMode. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR and InterpolationMode.BICUBIC are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable. Defaults to 'bilinear'.
    """

    def __init__(
        self,
        size: Union[int, Iterable[int]],
        scale: Iterable[float] = [0.08, 1.0],
        ratio: Iterable[float] = [3/4, 4/3],
        interpolation: Union[str, InterpolationMode] = 'bilinear',
        **kwargs
    ) -> None:
        if type(interpolation) is str:
            interpolation = _INTERPOLATION[interpolation]
        super().__init__(size, scale=scale, ratio=ratio,
                         interpolation=interpolation, **kwargs)
