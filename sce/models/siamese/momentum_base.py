# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import hydra
import math
from abc import ABC
from omegaconf import DictConfig
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn, Tensor
from torch.nn import Module

from sce.models.siamese.base import SiameseBaseModel


class MomentumSiameseBaseModel(SiameseBaseModel, ABC):
    """Abstract class to represent siamese models with a momentum branch.

    Subclasses should implement training_step method.

    Args:
        trunk (DictConfig): Config to build a trunk.
        optimizer (DictConfig): Config to build optimizers and schedulers.
        projector (Optional[DictConfig]): Config to build a project. Defaults to None.
        predictor (Optional[DictConfig]): Config to build a predictor. Defaults to None.
        transform (Optional[DictConfig]): Config to perform transformation on input. Defaults to None.
        normalize_outputs (bool): If true, normalize outputs. Defaults to True.
        num_global_crops (int): Number of global crops which are the first elements of each batch. Defaults to 2. 
        num_local_crops (int): Number of local crops which are the last elements of each batch. Defaults to 0.
        initial_momentum (float): initial value for the momentum update. Defaults to 0.996.
        scheduler_momentum (str): rule to update the momentum value. Defaults to cosine.
    """

    def __init__(
        self,
        trunk: DictConfig,
        optimizer: DictConfig,
        projector: Optional[DictConfig] = None,
        predictor: Optional[DictConfig] = None,
        transform: Optional[DictConfig] = None,
        normalize_outputs: bool = True,
        num_global_crops: int = 2,
        num_local_crops: int = 0,
        initial_momentum: int = 0.996,
        scheduler_momentum: str = 'cosine'
    ) -> None:
        super().__init__(trunk, optimizer, projector=projector, predictor=predictor, transform=transform,
                         normalize_outputs=normalize_outputs, num_global_crops=num_global_crops, num_local_crops=num_local_crops,)

        self.save_hyperparameters()
        self.momentum_trunk = hydra.utils.instantiate(trunk)
        self.momentum_projector = hydra.utils.instantiate(projector)

        self.initial_momentum = initial_momentum
        self.scheduler_momentum = scheduler_momentum
        self.current_momentum = initial_momentum

        for param in self.momentum_trunk.parameters():
            param.requires_grad = False

        if self.momentum_projector is not None:
            for param in self.momentum_projector.parameters():
                param.requires_grad = False

    def _update_momentum(self) -> float:
        if self.scheduler_momentum == 'constant':
            return self.current_momentum
        # Cosine rule that increase value from initial value to 1.
        elif self.scheduler_momentum == 'cosine':
            max_steps = self.training_steps_per_epoch * \
                self.trainer.max_epochs - 1  # -1 because self.global_step starts at 0
            momentum = 1 - (1 - self.initial_momentum) * (math.cos(math.pi *
                                                                   self.global_step / max_steps) + 1) / 2
            return momentum
        elif self.scheduler_momentum == 'cosine_epoch':
            # -1 because self.trainer.current_epoch starts at 0
            max_steps = self.trainer.max_epochs - 1
            momentum = 1 - (1 - self.initial_momentum) * (math.cos(math.pi *
                                                                   self.current_epoch / max_steps) + 1) / 2
            return momentum
        else:
            raise NotImplementedError(
                f'{self.scheduler_momentum} is not supported.')

    @torch.no_grad()
    def _update_weights(
        self,
        online: Union[Module, Tensor],
        target: Union[Module, Tensor]
    ) -> None:
        for (_, online_p), (_, target_p) in zip(
            online.named_parameters(),
            target.named_parameters(),
        ):
            target_p.data = self.current_momentum * target_p.data + \
                (1 - self.current_momentum) * online_p.data

    @torch.no_grad()
    def multi_crop_momentum_shared_step(self, x: List[Tensor]) -> Dict[str, Tensor]:
        h = self.trunk(torch.cat(x))
        z = self.projector(h) if self.projector is not None else h

        if self.hparams.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {'h': h, 'z': z}
    
    @torch.no_grad()
    def momentum_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step that pass the input tensor in the momentum trunk and momentum projector.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        h = self.momentum_trunk(x)
        z = self.momentum_projector(
            h) if self.momentum_projector is not None else h
        if self.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {'h': h, 'z': z}

    def on_train_batch_end(
        self,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        self._update_weights(self.trunk, self.momentum_trunk)
        if self.projector is not None:
            self._update_weights(self.projector, self.momentum_projector)

        # log momentum value used to update the weights
        self.log('pretrain/momentum_value', self.current_momentum,
                 on_step=True, on_epoch=False)

        # update momentum value
        self.current_momentum = self._update_momentum()
