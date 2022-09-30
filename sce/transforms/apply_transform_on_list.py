# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from PIL.Image import Image
from typing import Any, Iterable, List, Union
from torch import Tensor


class ApplyTransformOnList(object):
    """Apply transform to a list of input.

    Args:
        transform (Any, List[Any]): A transform for the list of input.
        list_len (int): len of the input. Defaults to 2.

    Example::

        transform = [
            ColorJitter()
        ]

        list_transform = ApplyTransformOnList(
            transform
        )
    """

    def __init__(
        self,
        transform: Any,
        list_len: int = 2
    ) -> None:
        super().__init__()
            
        self.list_len = list_len
        self.transform = transform

    def __call__(self, X: Iterable[Union[Image, Tensor]]) -> Tensor:
        X = [self.transform(X[i]) for i in range(self.list_len)]
        return X

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + \
            f'(transform={self.transform})'
        return format_string

class ApplyTransformsOnList(object):
    """Apply transform to a list of input.

    Args:
        transform (Any, List[Any]): A transform for the list of input.
        list_len (int): len of the input. Defaults to 2.

    Example::

        transform = [
            ColorJitter()
        ]

        list_transform = ApplyTransformOnList(
            transform
        )
    """

    def __init__(
        self,
        transforms: List[Any],
    ) -> None:
        super().__init__()
            
        self.list_len = len(transforms)
        self.transforms = transforms

    def __call__(self, X: Iterable[Union[Image, Tensor]]) -> Tensor:
        X = [self.transforms[i](X[i]) for i in range(self.list_len)]
        return X

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + \
            f'(transforms={self.transforms})'
        return format_string
