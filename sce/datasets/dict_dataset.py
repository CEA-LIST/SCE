# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Any, Iterable, Mapping, Union, Tuple
from PIL.Image import Image

from torch.utils.data import Dataset
from torch import Tensor


class DictDataset(Dataset):
    """Wrapper around a Dataset to have a dictionary as input for models.

    Args:
        dataset (Dataset): dataset to wrap around.

    Examples:
        dataset = ...
        dict_dataset = DictDataset(dataset)
    """

    def __init__(
        self,
        dataset: Dataset
    ) -> None:
        super().__init__()
        self.source_dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[int, Union[Image, Tensor], Any]:
        super_output = self.source_dataset[idx]
        if isinstance(super_output, Mapping):
            if "idx" in super_output:
                return super_output
            else:
                super_output["idx"] = idx
                return super_output
        elif isinstance(super_output, Iterable):
            if len(super_output) == 1:
                return {
                    "input": super_output[0],
                    "idx": idx
                } 
            elif len(super_output) == 2:
                return {
                    "input": super_output[0],
                    "label": super_output[1],
                    "idx": idx
                }
            else:
                raise NotImplementedError("Impossible to know what is in the list of super_ouput.")
        else:
            return {
                    "input": super_output,
                    "idx": idx
                } 

    def __len__(self):
        return len(self.source_dataset)
