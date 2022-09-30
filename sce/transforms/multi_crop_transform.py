# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from PIL.Image import Image
from typing import Any, List, Union

from torch import Tensor


class MultiCropTransform(object):
    """Define multi crop transform that apply several sets of transform to the inputs.

    Args:
        set_transforms (Mapping[Any, Any]): Dictionnary of sets of transforms specifying transforms and number of views per set

    Example::

        set_transforms = [
            {'transform': [...], 'num_views': ...},
            {'transform': [...], 'num_views': ...},
            ...
        ]

        transform = MultiCropTransform(
            set_transforms
        )
    """

    def __init__(
        self,
        set_transforms: List[Any]
    ) -> None:
        super().__init__()

        self.set_transforms = set_transforms
        transforms = []
        for set_transform in self.set_transforms:
            transforms.extend([set_transform['transform']]
                              * set_transform['num_views'])
        self.transforms = transforms

    def __call__(self, img: Union[Image, Tensor]) -> Tensor:
        transformed_images = [transform(img) for transform in self.transforms]
        return transformed_images

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('

        for set_transform in self.set_transforms:
            format_string += '\n'
            format_string += '    num views={0}\n'.format(
                set_transform['num_views'])
            format_string += '    transforms={0}'.format(
                set_transform['transform'])
            format_string += '\n)'
        return format_string