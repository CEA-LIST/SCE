# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from torch import Tensor
from torchvision.transforms import InterpolationMode

_MEANS = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar100': [0.5071, 0.4865, 0.4409],
    'stl10': [0.4914, 0.4822, 0.4465],
    'imagenet': [0.485, 0.456, 0.406]
}

_STDS = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'cifar100': [0.2009, 0.1984, 0.2023],
    'stl10': [0.2471, 0.2435, 0.2616],
    'imagenet': [0.229, 0.224, 0.225]
}

_INTERPOLATION = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
}


class ToDevice(object):
    """Place on device tensor.

    Args:
        device (str): Device. Defaults to cpu.
    """

    def __init__(self, device: str = 'cpu') -> None:
        self.device = device

    def __call__(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        return x

    def __repr__(self) -> str:
        return f"{__class__.__name__}(device={self.device})"


class Squeeze(object):
    """Squeeze tensor."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        x = x.squeeze()
        return x

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"


class Unsqueeze(object):
    """
    Unsqueeze tensor.
    
    Args:
        dim (int): Dimension to unsqueeze. Defaults to 0.
    """

    def __init__(self, dim: int=0) -> None:
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=self.dim)
        return x

    def __repr__(self) -> str:
        return f"{__class__.__name__}(dim={self.dim})"
