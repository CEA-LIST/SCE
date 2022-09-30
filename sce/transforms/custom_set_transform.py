# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import List

from torch.nn import Module
from torch import Tensor


class CustomSetTransform(Module):
    """Define a custom transform based on list of transforms.

    Args:
        transforms (List[Module]): List of transforms.

    Example::

        transforms = [...]
        custom_transform = CustomSetTransform(transforms)
    """

    def __init__(
        self,
        transforms: List[Module]
    ) -> None:
        super().__init__()

        self.transforms = transforms

    def forward(self, img: Tensor):
        for transform in self.transforms:
            img = transform(img)

        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('

        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
