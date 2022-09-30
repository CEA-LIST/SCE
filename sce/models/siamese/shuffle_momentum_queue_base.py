# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

"""
References:
    - SCE: https://arxiv.org/pdf/2111.14585.pdf
"""

from abc import ABC
from omegaconf import DictConfig
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn, Tensor

from sce.models.modules.gather import concat_all_gather_without_backprop
from sce.models.siamese.shuffle_momentum_base import ShuffleMomentumSiameseBaseModel


class ShuffleMomentumQueueBaseModel(ShuffleMomentumSiameseBaseModel, ABC):
    """ShuffleMomentumQueueBaseModel model.

    Args:
        trunk (DictConfig): Config tu build a trunk.
        optimizer (DictConfig): Config tu build optimizers and schedulers.
        projector (Optional[DictConfig]): Config to build a project. Defaults to None.
        predictor (Optional[DictConfig]): Config to build a predictor. Defaults to None.
        transform (Optional[DictConfig]): Config to perform transformation on input. Defaults to None.
        normalize_outputs (bool): If true, normalize outputs. Defaults to True.
        num_global_crops (int): Number of global crops which are the first elements of each batch. Defaults to 2. 
        num_local_crops (int): Number of local crops which are the last elements of each batch. Defaults to 0. 
        initial_momentum (float): initial value for the momentum update. Defaults to 0.999.
        scheduler_momentum (str): rule to update the momentum value. Defaults to constant.
        shuffle_bn (bool): If true, apply shuffle normalization trick from MoCo. Defaults to True.
        num_devices (int): Number of devices used to train the model in each node. Defaults to 1.
        simulate_n_devices (int): Number of devices to simulate to apply shuffle trick. Requires shuffle_bn to be True and num_devices to be 1. Defaults to 8.
        queue (Optional[DictConfig]): Config to build a queue. Defaults to None.
        sym (bool): If true, symmetrised the loss. Defaults to False.
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
        initial_momentum: int = 0.999,
        scheduler_momentum: str = 'constant',
        shuffle_bn: bool = True,
        num_devices: int = 1,
        simulate_n_devices: int = 8,
        queue: Optional[DictConfig] = None,
        sym: bool = False,
    ) -> None:
        super().__init__(trunk, optimizer, projector=projector, predictor=predictor, transform=transform, normalize_outputs=normalize_outputs,
                         num_global_crops=num_global_crops, num_local_crops=num_local_crops, initial_momentum=initial_momentum, scheduler_momentum=scheduler_momentum, shuffle_bn=shuffle_bn, num_devices=num_devices, simulate_n_devices=simulate_n_devices)

        self.save_hyperparameters()

        self.sym = sym

        if queue is not None:
            self.register_buffer('queue', torch.randn(
                queue.feature_dim, queue.size))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.queue_size = queue.size
            self.queue_feature_dim = queue.feature_dim
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        else:
            raise NotImplementedError("SCE requires a queue to work.")

    @torch.no_grad()
    def update_queue(self, x: Tensor) -> None:
        batch_size = x.shape[0]

        ptr = int(self.queue_ptr)
        # for simplicity
        assert self.queue_size % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = x.T

        # move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


    def training_step(self, batch: Iterable[Any], batch_idx: int) -> Dict[str, Tensor]:
        X = batch["input"]
        X = [X] if isinstance(X, Tensor) else X

        assert len(X) == self.num_crops

        if self.transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    X = self.transform(X)

        if self.sym:
            outs_online = [self.shared_step(x) for x in X]
            outs_momentum = [self.momentum_shared_step(
                x) for x in X[:self.num_global_crops]]

            tot_loss = 0
            for i in range(self.num_global_crops):
                for j in range(self.num_crops):
                    if i == j:
                        continue
                    loss = self.compute_loss(
                        outs_online[j]['q'], outs_momentum[i]['z'])
                    tot_loss += loss
            tot_loss /= (self.num_crops - 1) * self.num_global_crops

            k_targets = [concat_all_gather_without_backprop(
                outs_momentum[i]['z']) for i in range(self.num_global_crops)]
            k_target = torch.cat(k_targets, dim=0)
            self.update_queue(k_target)

        else:
            outs_online = [self.shared_step(
                x) for x in X[1:self.num_crops]]
            outs_momentum = self.momentum_shared_step(X[0])

            tot_loss = 0
            for j in range(self.num_crops - 1):
                loss = self.compute_loss(
                    outs_online[j]['q'], outs_momentum['z'])
                tot_loss += loss
            tot_loss /= self.num_crops - 1

            k_target = concat_all_gather_without_backprop(
                outs_momentum['z'])

            self.update_queue(k_target)

        outputs = {'loss': tot_loss}
        # Only keep outputs from first computation to avoid unnecessary time and memory cost.
        outputs.update(outs_online[0])
        for name_output, output in outputs.items():
            if name_output != 'loss':
                outputs[name_output] = output.detach()

        self.log('pretrain/loss', outputs['loss'],
                 on_step=True, on_epoch=True)

        return outputs
