# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Iterable, Union

from torchvision.transforms import ColorJitter, RandomApply


class RandomColorJitter(RandomApply):
    """Randomly apply color jitter to a tensor image with a certain probability.

    Args:
        brightness (Union[float, Iterable[float]]): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers. Defaults to 0.4.
        contrast (Union[float, Iterable[float]]): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers. Defaults to 0.4.
        saturation (Union[float, Iterable[float]]): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers. Defaults to 0.4.
        hue (Union[float, Iterable[float]]): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. Defaults to 0.1.
        p (float): Probability to apply color jitter. Defaults to 0.8.

    Example::

        transform = RandomColorJitter()
    """

    def __init__(
        self,
        brightness: Union[float, Iterable[float]] = 0.4,
        contrast: Union[float, Iterable[float]] = 0.4,
        saturation: Union[float, Iterable[float]] = 0.4,
        hue: Union[float, Iterable[float]] = 0.1,
        p: float = 0.8
    ) -> None:

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        color_jitter = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

        super().__init__(transforms=[color_jitter], p=p)
