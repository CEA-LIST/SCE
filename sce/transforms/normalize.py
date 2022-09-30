# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Iterable, Optional
from torchvision.transforms import Normalize

from sce.transforms.utils import _MEANS, _STDS


class Normalize(Normalize):
    """Normalize a tensor image from a dataset with mean and standard deviation.

    Args:
        dataset (Optional[str]): Name of the dataset to retrieve the mean and standard deviation. Defaults to None.
        mean (Optional[float]): Mean of dataset inputs. Defaults to None.
        std (Optional[float]): Std of dataset inputs. Defaults to None.

    Example::

        transform = Normalize(dataset="imagenet", mean=None, std=None)
        transform = Normalize(dataset=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """

    def __init__(
        self,
        dataset: Optional[str] = None,
        mean: Optional[Iterable[float]] = None,
        std: Optional[Iterable[float]] = None
    ) -> None:

        assert not (dataset is None and (mean is None or std is None)
                    ), "You should provide a dataset name or mean and std to normalize"

        if dataset is not None:
            mean = _MEANS[dataset]
            std = _STDS[dataset]
        else:
            assert mean is not None and std is not None, "If you do not provide a dataset name, you should provide mean and std"
        super().__init__(mean=mean, std=std)
