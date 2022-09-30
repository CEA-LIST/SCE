# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from omegaconf import DictConfig
from typing import Any, Dict, Optional

from torch.optim import Optimizer
from sce.optimizers.utils import scale_learning_rate

from sce.schedulers.utils import _SCHEDULERS


def scheduler_factory(
    optimizer: Optimizer,
    name: str,
    params: DictConfig = {},
    interval: str = 'epoch',
    num_steps_per_epoch: Optional[int] = None,
    scaler: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Scheduler factory.

    Args:
        optimizer (Optimizer): Optimizer to wrap around.
        name (str): Name of the scheduler to retrieve the scheduler constructor from the _SCHEDULERS dict.
        params (DictConfig): Scheduler parameters for the scheduler constructor. Defaults to {}.
        interval (str): Interval to call step, if epoch call `.step()` at each epoch. Defaults to 'epoch'.
        num_steps_per_epoch (Optional[int]): Number of steps per epoch. Usefull for some schedulers. Defaults to None.
        scaler (Optional[str]): Scaler rule for the initial learning rate. Defaults to None.
        batch_size (Optional[int]): Batch size for the input of the model. Defaults to None.

    Returns:
        Dict[str, Any]: Scheduler configuration for pytorch lightning.
    """

    if interval == 'step':
        if name == 'linear_warmup_cosine_annealing_lr':
            params.max_epochs = num_steps_per_epoch * params.max_epochs
            params.warmup_epochs = num_steps_per_epoch * params.warmup_epochs
            if params.get('eta_min'):
                params.eta_min = scale_learning_rate(params.eta_min, scaler, batch_size)
        elif name == 'cosine_annealing_lr':
            if params.get('eta_min'):
                params.eta_min = scale_learning_rate(params.eta_min, scaler, batch_size)

    scheduler = _SCHEDULERS[name](optimizer=optimizer, **params)

    return {'scheduler': scheduler, 'interval': interval}
