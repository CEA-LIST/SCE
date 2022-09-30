# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import hydra
from omegaconf import DictConfig
from typing import Optional, Tuple
from pytorch_lightning.utilities import rank_zero_info

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from sce.optimizers.utils import _OPTIMIZERS, _NORM_LAYERS, filter_learnable_params, retrieve_model_params, scale_learning_rate


def optimizer_factory(
    name: str,
    initial_lr: float,
    model: Module,
    batch_size: Optional[int] = None,
    num_steps_per_epoch: Optional[int] = None,
    exclude_wd_norm: bool = False,
    exclude_wd_bias: bool = False,
    scaler: Optional[str] = None,
    params: DictConfig = {},
    divide_wd_by_lr: bool = False,
    scheduler: Optional[DictConfig] = None,
) -> Tuple[Optimizer, Optional[_LRScheduler]]:
    """Optimizer factory to build optimizers and optionally an attached scheduler.

    Args:
        name (str): Name of the scheduler to retrieve the optimizer constructor from _OPTIMIZERS dict.
        initial_lr (float): Initial learning rate.
        model (Module): Model to optimize.
        batch_size (Optional[int]): Batch size for the input of the model. Defaults to None.
        num_steps_per_epoch (Optional[int]): Number of steps per epoch. Usefull for some schedulers. Defaults to None.
        exclude_wd_norm (bool): If true, exclude normalization layers to be regularized by weight decay. Defaults to False.
        exclude_wd_bias (bool): If true, exclude bias layers to be regularized by weight decay. Defaults to False.
        scaler (Optional[str]): Scaler rule for the initial learning rate. Defaults to None.
        params (DictConfig): Parameters for the optimizer constructor. Defaults to {}.
        divide_wd_by_lr (bool): If True, divide the weight decay by the value of the learning rate. Defaults to False.
        scheduler (Optional[DictConfig]): Scheduler config. Defaults to None.

    Returns:
        Tuple[Optimizer, Optional[_LRScheduler]]: Optimizer with its optional scheduler.
    """

    optimizer_class = _OPTIMIZERS[name]

    lr = scale_learning_rate(initial_lr, scaler, batch_size)

    if "weight_decay" in params and divide_wd_by_lr:
        params["weight_decay"] /= lr
        rank_zero_info(
            f"weight_decay has been scaled to {params['weight_decay']}"
        )

    modules_without_decay = []
    keys_without_decay = []

    if exclude_wd_norm:
        modules_without_decay.extend(_NORM_LAYERS)
    if exclude_wd_bias:
        keys_without_decay.append('bias')

    # Retrieve all the parameters in the model excluding the specified modules and keys.
    no_wd_parameters, wd_parameters = retrieve_model_params(
        model, modules_without_decay, keys_without_decay)

    # Filter learnable params as a property of the model if it is defined.
    wd_parameters = filter_learnable_params(wd_parameters, model)
    no_wd_parameters = filter_learnable_params(no_wd_parameters, model)

    # If the group of parameters without weight decay is not empty
    if no_wd_parameters != []:
        optimizer = optimizer_class([
            {'params': wd_parameters},
            {'params': no_wd_parameters, 'weight_decay': 0.}],
            lr=lr, **params
        )
    else:
        optimizer = optimizer_class(wd_parameters, lr=lr, **params)
    
    rank_zero_info(f"{model._get_name()} optimizer's: filter_wd={len(no_wd_parameters)}, non_filtered wd = {len(wd_parameters)}")

    if scheduler is not None:
        scheduler = hydra.utils.instantiate(
            scheduler, num_steps_per_epoch=num_steps_per_epoch, optimizer=optimizer, scaler=scaler, batch_size=batch_size)

    return optimizer, scheduler
