# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from torchvision.transforms import Compose, RandomApply

from sce.transforms.apply_key import ApplyTransformInputKeyOnList, ApplyTransformOnDict, ApplyTransformInputKey
from sce.transforms.apply_transform_on_list import ApplyTransformOnList, ApplyTransformsOnList
from sce.transforms.contrastive_transform import ContrastiveTransform, GPUImageContrastiveTransform
from sce.transforms.custom_set_transform import CustomSetTransform
from sce.transforms.dict_keep_keys import DictKeepInputLabel, DictKeepInputLabelIdx, DictKeepKeys
from sce.transforms.dict_to_list_from_keys import DictToListFromKeys, DictToListInputLabel
from sce.transforms.linear_classifier_transform import LinearClassifierSmallTrainTransform, LinearClassifierTrainTransform, LinearClassifierValTransform
from sce.transforms.logistic_regression_transform import LogisticRegressionClassifierTransform
from sce.transforms.multi_crop_transform import MultiCropTransform
from sce.transforms.normalize import Normalize
from sce.transforms.only_input_transform import OnlyInputListTransform, OnlyInputTransform, OnlyInputTransformWithDictTransform
from sce.transforms.random_color_jitter import RandomColorJitter
from sce.transforms.random_gaussian_blur import RandomGaussianBlur, PILGaussianBlur, RandomPILGaussianBlur
from sce.transforms.random_resized_crop import RandomResizedCrop
from sce.transforms.random_solarize import GPURandomSolarize
from sce.transforms.resize import Resize
from sce.transforms.utils import ToDevice, Squeeze, Unsqueeze

from kornia.augmentation import ColorJitter as GPURandomColorJitter
from kornia.augmentation import RandomCrop as GPURandomResizedCrop
from kornia.augmentation import Normalize as GPUNormalize
from kornia.augmentation import RandomGaussianBlur as GPURandomGaussianBlur
from kornia.augmentation import RandomGrayscale as GPURandomGrayscale
from kornia.augmentation import RandomHorizontalFlip as GPURandomHorizontalFlip

from torchvision.transforms import CenterCrop, RandomCrop, RandomGrayscale, RandomHorizontalFlip, RandomSolarize, ToPILImage, ToTensor
