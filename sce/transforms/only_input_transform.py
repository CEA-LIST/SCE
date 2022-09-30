# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Callable

from torchvision.transforms import Compose

from sce.transforms import ApplyTransformInputKeyOnList, ApplyTransformInputKey, ApplyTransformOnDict, DictKeepInputLabelIdx


class OnlyInputListTransform(Compose):
    def __init__(self, transform: Callable) -> None:
        transforms = [ApplyTransformInputKeyOnList(
            transform), DictKeepInputLabelIdx()]

        super().__init__(transforms=transforms)


class OnlyInputTransform(Compose):
    def __init__(self, transform: Callable) -> None:
        transforms = [ApplyTransformInputKey(
            transform), DictKeepInputLabelIdx()]

        super().__init__(transforms=transforms)


class OnlyInputTransformWithDictTransform(Compose):
    def __init__(self, transform: Callable, dict_transform: Callable, first_dict: bool = False) -> None:
        if first_dict:
            transforms = [ApplyTransformInputKey(
                transform), ApplyTransformOnDict(dict_transform), DictKeepInputLabelIdx()]
        else:
            transforms = [ApplyTransformInputKey(
                transform), ApplyTransformOnDict(dict_transform), DictKeepInputLabelIdx()]

        super().__init__(transforms=transforms)
