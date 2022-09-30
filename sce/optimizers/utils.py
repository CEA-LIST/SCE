# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import numpy as np
from typing import Iterable, List, Optional, Tuple

from pytorch_lightning.utilities import rank_zero_info
import torch
from torch.nn import Module, BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, SyncBatchNorm
from torch.nn.parameter import Parameter

from sce.models.modules.split_batch_norm import SplitBatchNorm2D
from sce.optimizers.lars import LARS

_NORM_LAYERS = (BatchNorm1d, BatchNorm2d,
                BatchNorm3d, SyncBatchNorm,
                SplitBatchNorm2D, LayerNorm)
_OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'lars': LARS
}


def retrieve_model_params(
    model: Module,
    modules_to_filter: Iterable[Module] = [],
    keys_to_filter: Iterable[str] = []
) -> Tuple[List[Parameter], List[Parameter]]:
    """
    Retrieve sets of filtered and not filtered parameters from a model.

    Args:
        model (Module): Model to retrieve the params from.
        modules_to_filter (List[Module]): Module to filter.
        keys_to_filter (List[str]): keys to filter.

    Returns:
        Tuple[List[Parameter], List[Parameter]]: Filtered parameters, other parameters.
    """
    other_parameters = []
    filtered_parameters = []
    for module in model.modules():
        if type(module) in modules_to_filter:
            for param_name, param in module.named_parameters(recurse=False):
                filtered_parameters.append(param)
        else:
            for param_name, param in module.named_parameters(recurse=False):
                no_key = all(
                    [param_name != key for key in keys_to_filter])
                if no_key:
                    other_parameters.append(param)
                else:
                    filtered_parameters.append(param)
    return filtered_parameters, other_parameters


def filter_learnable_params(
    parameters: Iterable[Parameter],
    model: Module
) -> List[Parameter]:
    """Filter passed parameters to be in learnable parameters list from model. If model do not have learnable_params property defined, return all passed parameters.

    Args:
        parameters (Iterable[Parameter]): Parameters to filter.
        model (Module): Model to retrieve learnable parameters from.

    Returns:
        List[Parameter]: Learnable parameters.
    """
    if hasattr(model, 'learnable_params'):
        return [param for param in parameters if any(
            [param is learnable_param for learnable_param in model.learnable_params])]
    else:
        rank_zero_info(
            f'Model of type {type(model)} has no learnable parameters defined, all passed parameters returned.')
        return list(parameters)


def scale_learning_rate(
    initial_lr: int,
    scaler: Optional[str] = None,
    batch_size: Optional[int] = None
) -> int:
    """Scale the initial learning rate.

    Args:
        initial_lr (int): Initial learning rate.
        scaler (Optional[str]): Scaler rule. Defaults to None.
        batch_size (Optional[int]): Batch size to scale the learning rate. Defaults to None.

    Returns:
        int: Scaled initial learning rate.
    """
    if scaler is None or scaler == 'none':
        return initial_lr
    elif scaler == 'linear':
        lr = initial_lr * batch_size / 256
    elif scaler == 'sqrt':
        lr = initial_lr * np.sqrt(batch_size)
    return lr
