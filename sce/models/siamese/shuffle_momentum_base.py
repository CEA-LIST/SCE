# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from abc import ABC
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn

import torch
from torch import nn, Tensor

from sce.models.modules.gather import concat_all_gather_without_backprop
from sce.models.modules.split_batch_norm import convert_to_split_batchnorm
from sce.models.siamese.momentum_base import MomentumSiameseBaseModel
from sce.utils.strategies import get_num_devices_in_trainer


class ShuffleMomentumSiameseBaseModel(MomentumSiameseBaseModel, ABC):
    """Abstract class to represent siamese models with a momentum branch and possibility to shuffle input elements in momentum branch to apply normalization trick from MoCo.

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
        shuffle_bn (bool): If true, apply shuffle normalization trick from MoCo. Defaults to True.
        num_devices (int): Number of devices used to train the model in each node. Defaults to 1.
        simulate_n_devices (int): Number of devices to simulate to apply shuffle trick. Requires shuffle_bn to be True and num_devices to be 1. Defaults to 8.
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
        scheduler_momentum: str = 'cosine',
        shuffle_bn: bool = True,
        num_devices: int = 1,
        simulate_n_devices: int = 8
    ) -> None:
        super().__init__(trunk, optimizer, projector=projector, predictor=predictor, transform=transform, normalize_outputs=normalize_outputs,
                         num_global_crops=num_global_crops, num_local_crops=num_local_crops, initial_momentum=initial_momentum, scheduler_momentum=scheduler_momentum,)

        self.save_hyperparameters()

        self.num_devices = num_devices
        self.shuffle_bn = shuffle_bn
        self.simulate_n_devices = simulate_n_devices

        if self.num_devices == -1 and self.shuffle_bn:
            rank_zero_info(
                f'In {__class__.__name__} when num_devices=-1, it is assumed that there are more than one device.')

        elif self.num_devices <= 1 and self.shuffle_bn:
            if self.simulate_n_devices <= 1:
                AttributeError(
                    'if num_devices is 1 and shuffle_bn is True, the simulate_n_devices attribute should be superior to 1')
            self.momentum_trunk = convert_to_split_batchnorm(
                self.momentum_trunk, self.simulate_n_devices)
            if self.momentum_projector is not None:
                self.momentum_projector = convert_to_split_batchnorm(
                    self.momentum_projector, self.simulate_n_devices)

    def on_train_start(self):
        old_num_devices = self.num_devices
        self.num_devices = get_num_devices_in_trainer(self.trainer)
        if old_num_devices != self.num_devices:
            rank_zero_warn(
                f'Num devices passed to {__class__.__name__}: {old_num_devices} has been updated to {self.num_devices}.')

    @torch.no_grad()
    def momentum_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step that call either '_momentum_shared_step_n_devices' or '_momentum_shared_step_single_device' depending the number of devices used for training.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        if self.num_devices > 1:
            return self._momentum_shared_step_n_devices(x)

        else:
            return self._momentum_shared_step_single_device(x)

    @torch.no_grad()
    def _momentum_shared_step_n_devices(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step with several devices passing input tensor in momentum trunk and momentum projector. If shuffle_bn is True, it gathers and shuffles the input tensors across devices following MoCo batch norm trick.  

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        if self.shuffle_bn:
            x, idx_unshuffle = self._batch_shuffle_ddp(x)

        h = self.momentum_trunk(x)
        z = self.momentum_projector(
            h) if self.momentum_projector is not None else h

        if self.shuffle_bn:
            z = self._batch_unshuffle_ddp(z, idx_unshuffle)

        if self.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {'h': h, 'z': z}

    @torch.no_grad()
    def _momentum_shared_step_single_device(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step with one device passing input tensor in momentum trunk and momentum projector. If shuffle_bn is True, it shuffles the input tensor across device following MoCo batch norm trick which is simulated in the trunk.  

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        if self.shuffle_bn:
            x, idx_unshuffle = self._batch_shuffle_single_device(x)

        h = self.momentum_trunk(x)
        z = self.momentum_projector(
            h) if self.momentum_projector is not None else h

        if self.shuffle_bn:
            z = self._batch_unshuffle_single_device(z, idx_unshuffle)

        if self.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {'h': h, 'z': z}

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Unshuffle the shuffled tensor along first dimension across devices.

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x (Tensor): The shuffled tensor.
            idx_unshuffle (Tensor): The unshuffle indices to retrieve original tensor before its shuffling.

        Returns:
            Tuple[Tensor, Tensor]: The shuffled tensor and the unshuffle indices.
        """
        # gather from all devices
        x_gather = concat_all_gather_without_backprop(x)
        batch_size_all = x_gather.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=self.device)

        # broadcast to all devices
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this device
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(self.num_devices, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: Tensor, idx_unshuffle: Tensor) -> Tensor:
        """Unshuffle the shuffled tensor along first dimension across devices.

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x (Tensor): The shuffled tensor.
            idx_unshuffle (Tensor): The unshuffle indices to retrieve original tensor before its shuffling.

        Returns:
            Tuple[Tensor, Tensor]: The shuffled tensor and the unshuffle indices.
        """
        # gather from all devices
        x_gather = concat_all_gather_without_backprop(x)

        # restored index for this device
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(self.num_devices, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _batch_shuffle_single_device(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Shuffle the input tensor along first dimension on current device.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tuple[Tensor, Tensor]: The shuffled tensor and the unshuffle indices.
        """
        # random shuffle index
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size, device=self.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_device(self, x: Tensor, idx_unshuffle: Tensor) -> Tensor:
        """Unshuffle the shuffled tensor along first dimension on current device.

        Args:
            x (Tensor): The shuffled tensor.
            idx_unshuffle (Tensor): The unshuffle indices to retrieve original tensor before its shuffling.

        Returns:
            Tuple[Tensor, Tensor]: The shuffled tensor and the unshuffle indices.
        """
        return x[idx_unshuffle]
