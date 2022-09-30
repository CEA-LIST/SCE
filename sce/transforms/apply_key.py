# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Callable, Dict, List

import torch
from torch import Tensor


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='input',
        >>>       transform=CenterCrop(256)),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class ApplyTransformToKeyOnList:
    """
    Applies transform to key of dictionary input wherein input is a list
    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied
    Example:
         >>>  transforms.ApplyTransformToKeyOnList(
        >>>       key='input',
        >>>       transform=CenterCrop(256),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable) -> None:  # pyre-ignore[24]
        self._key = key
        self._transform = transform

    def __call__(
        self, x: Dict[str, List[Tensor]]
    ) -> Dict[str, List[Tensor]]:
        x[self._key] = [self._transform(a) for a in x[self._key]]
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}(key={self._key}, transform={self._transform})"


class ApplyTransformInputKeyOnList(ApplyTransformToKeyOnList):
    """
    Apply Transform to the input key.

    Args:
        transform (Callable): The transform to apply.
    """

    def __init__(self, transform: Callable):
        super().__init__('input', transform=transform)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"


class ApplyTransformInputKey(ApplyTransformToKey):
    """
    Apply Transform to the input key.

    Args:
        transform (Callable): The transform to apply.
    """

    def __init__(self, transform: Callable):
        super().__init__('input', transform=transform)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"


class ApplyTransformOnDict():
    """
    Apply Transform to the audio key.

    Args:
        transform (Callable): The transform to apply.
    """

    def __init__(self, transform: Callable):
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._transform(x)
        
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"
