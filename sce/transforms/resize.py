# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Iterable, Union

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from sce.transforms.utils import _INTERPOLATION


class Resize(transforms.Resize):
    """Resize the input image to the given size. If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (Union[int, Iterable[int]]): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
            In torchscript mode size as single int is not supported, use a sequence of length 1: [size, ].
        interpolation (Union[str, InterpolationMode]): Desired interpolation enum defined by
            torchvision.transforms.InterpolationMode. If str type, lookup in _Interpolation dictionnary to retrieve the InterpolationMode. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR and InterpolationMode.BICUBIC are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable. Defaults to bilinear.
    """

    def __init__(
        self,
        size: Union[int, Iterable[int]],
        interpolation: Union[str, InterpolationMode] = 'bilinear',
    ) -> None:
        interpolation = _INTERPOLATION[interpolation]
        super().__init__(size, interpolation=interpolation)
