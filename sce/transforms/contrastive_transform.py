# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Optional
from kornia.augmentation import ImageSequential
from kornia.augmentation import ColorJitter as GPURandomColorJitter
from kornia.augmentation import RandomCrop as GPURandomResizedCrop
from kornia.augmentation import Normalize as GPUNormalize
from kornia.augmentation import RandomGaussianBlur as GPURandomGaussianBlur
from kornia.augmentation import RandomGrayscale as GPURandomGrayscale
from kornia.augmentation import RandomHorizontalFlip as GPURandomHorizontalFlip

from torchvision.transforms.transforms import Compose, RandomGrayscale, RandomHorizontalFlip, ToTensor, RandomSolarize

from sce.transforms.random_resized_crop import RandomResizedCrop
from sce.transforms.random_color_jitter import RandomColorJitter
from sce.transforms.random_gaussian_blur import RandomGaussianBlur
from sce.transforms.normalize import Normalize
from sce.transforms.random_solarize import GPURandomSolarize


class ContrastiveTransform(Compose):
    """Define a transform for self-supervised contrastive learning defined by SimCLR.

    Args:
        random_resized_crop (RandomResizedCrop): Random resized crop transform
        random_color_jitter (RandomColorJitter): Random color jitter transform
        random_grayscale (RandomGrayScale): Random grayscale transform
        random_horizontal_flip (RandomHorizontalFlip): Random horizontal flip transform
        random_gaussian_blur (RandomGaussianBlur): Random gaussian blur transform
        random_solarize (RandomSolarize): Random solarize transform
        to_tensor (ToTensor): To tensor transform
        Normalize (Normalize): Normalize transform

    Example::

        random_resized_crop = RandomResizedCrop(...)
        random_color_jitter = RandomColorJitter(...)
        random_grayscale = RandomGrayscale(...)
        random_horizontal_flip = RandomHorizontalFlip(...)
        random_gaussian_blur = RandomGaussianBlur(...)
        random_solarize = RandomSolarize(...)
        to_tensor = ToTensor(...)
        normalize = Normalize(...)

        contrastive_transform = ContrastiveTransform(
            random_resized_crop, random_color_jitter, random_grayscale,
            random_horizontal_flip, random_gaussian_blur, random_solarize, to_tensor,
            normalize
        )
    """

    def __init__(
        self,
        random_resized_crop: Optional[RandomResizedCrop] = None,
        random_color_jitter: Optional[RandomColorJitter] = None,
        random_grayscale: Optional[RandomGrayscale] = None,
        random_gaussian_blur: Optional[RandomGaussianBlur] = None,
        random_solarize: Optional[RandomSolarize] = None,
        random_horizontal_flip: Optional[RandomHorizontalFlip] = None,
        to_tensor: Optional[ToTensor] = None,
        normalize: Optional[Normalize] = None,
    ) -> None:
        transforms = [] 
        
        if random_resized_crop is not None:
            transforms += [random_resized_crop]
        
        if random_color_jitter is not None:
            transforms += [random_color_jitter]
        
        if random_grayscale is not None:
            transforms += [random_grayscale]
        
        if random_gaussian_blur is not None:
            transforms += [random_gaussian_blur]
                
        if random_solarize is not None:
            transforms += [random_solarize]
        
        if random_horizontal_flip is not None:
            transforms += [random_horizontal_flip]
        
        if to_tensor is not None:
            transforms += [to_tensor]
        
        if normalize is not None:
            transforms += [normalize]

        super().__init__(transforms)


class GPUImageContrastiveTransform(ImageSequential):
    """Define a transform for self-supervised contrastive learning defined by SimCLR.

    Args:
        random_resized_crop (GPURandomResizedCrop): Random resized crop transform
        random_color_jitter (GPURandomColorJitter): Random color jitter transform
        random_grayscale (GPURandomGrayScale): Random grayscale transform
        random_horizontal_flip (GPURandomHorizontalFlip): Random horizontal flip transform
        random_gaussian_blur (GPURandomGaussianBlur): Random gaussian blur transform
        random_solarize (GPURandomSolarize): Random solarize transform
        Normalize (GPUNormalize): Normalize transform

    Example::

        random_resized_crop = GPURandomResizedCrop(...)
        random_color_jitter = GPURandomColorJitter(...)
        random_grayscale = GPURandomGrayscale(...)
        random_horizontal_flip = GPURandomHorizontalFlip(...)
        random_gaussian_blur = GPURandomGaussianBlur(...)
        random_solarize = GPURandomSolarize(...)
        normalize = GPUNormalize(...)

        contrastive_transform = GPUImageContrastiveTransform(
            random_resized_crop, random_color_jitter, random_grayscale,
            random_horizontal_flip, random_gaussian_blur, random_solarize, normalize
        )
    """

    def __init__(
        self,
        random_resized_crop: Optional[GPURandomResizedCrop] = None,
        random_color_jitter: Optional[GPURandomColorJitter] = None,
        random_grayscale: Optional[GPURandomGrayscale] = None,
        random_gaussian_blur: Optional[GPURandomGaussianBlur] = None,
        random_solarize: Optional[GPURandomSolarize] = None,
        random_horizontal_flip: Optional[GPURandomHorizontalFlip] = None,
        normalize: Optional[GPUNormalize] = None,
        **kwargs
    ) -> None:
        transforms = [] 
        
        if random_resized_crop is not None:
            transforms += [random_resized_crop]
        
        if random_color_jitter is not None:
            transforms += [random_color_jitter]
        
        if random_grayscale is not None:
            transforms += [random_grayscale]
        
        if random_gaussian_blur is not None:
            transforms += [random_gaussian_blur]
                
        if random_solarize is not None:
            transforms += [random_solarize]
        
        if random_horizontal_flip is not None:
            transforms += [random_horizontal_flip]
        
        if normalize is not None:
            transforms += [normalize]

        super().__init__(*transforms, **kwargs)
