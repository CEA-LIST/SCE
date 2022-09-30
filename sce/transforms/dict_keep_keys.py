# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

"""
References:
    - https://github.com/kalyanvasudev/pytorchInput-1/blob/export-D33431232/pytorchInput_trainer/pytorchInput_trainer/datamodule/transforms.py
"""

from typing import Dict, List

from torch import Tensor


class DictKeepKeys(object):
    """
    Keep specified keys in dict.

    Args:
        transform (Callable): The transform to apply.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, x: Dict[str, Tensor]) -> List[Tensor]:
        x = {key: value for key, value in x.items() if key in self.keys}
        return x

    def __repr__(self):
        return f"{self.__class__.__name__ }(keys={self.keys})"


class DictKeepInputLabel(DictKeepKeys):
    """
    Transform dict to list containing values of only Input and label.

    Args:
        transform (Callable): The transform to apply.
    """

    def __init__(self):
        super().__init__(['input', 'label'])


class DictKeepInputLabelIdx(DictKeepKeys):
    """
    Transform dict to list containing values of only Input and label.

    Args:
        transform (Callable): The transform to apply.
    """

    def __init__(self):
        super().__init__(['input', 'label', 'idx', 'aug_index'])
