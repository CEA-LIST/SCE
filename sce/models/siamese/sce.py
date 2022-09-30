# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

"""
References:
    - SCE: https://arxiv.org/pdf/2111.14585.pdf
"""

from omegaconf import DictConfig
from typing import Dict, Optional

import torch
from torch import nn, Tensor

from sce.models.modules.gather import concat_all_gather_without_backprop
from sce.models.siamese.shuffle_momentum_queue_base import ShuffleMomentumQueueBaseModel
from sce.utils.utils import warmup_value

LARGE_NUM = 1e9


class SCEModel(ShuffleMomentumQueueBaseModel):
    """SCE model.

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
        temp (float): Temperature parameter to scale the online similarities. Defaults to 0.1.
        temp_m (float): Temperature parameter to scale the target similarities. Initial value if warmup applied. Defaults to 0.05.
        coeff (float): Coeff parameter between InfoNCE and relational aspects. Defaults to 0.5.
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
        temp: float = 0.1,
        temp_m: float = 0.05,
        coeff: float = 0.5
    ) -> None:
        super().__init__(trunk, optimizer, projector=projector, predictor=predictor, transform=transform, normalize_outputs=normalize_outputs,
                         num_global_crops=num_global_crops, num_local_crops=num_local_crops, initial_momentum=initial_momentum, scheduler_momentum=scheduler_momentum, shuffle_bn=shuffle_bn, num_devices=num_devices, simulate_n_devices=simulate_n_devices,
                         queue=queue, sym=sym)

        self.save_hyperparameters()

        self.temp = temp
        self.temp_m = temp_m

        self.coeff = coeff

    def compute_loss(self, q: Tensor, k: Tensor) -> Dict[str, Tensor]:
        """Compute the SCE loss.

        .. math::
            L_{SCE} = - \frac{1}{N} \sum_{i=1}^N\sum_{k=1}^N w^2_{ik} \log\left(p^1_{ik}\right)
            s^2_{ik} = \frac{{1}_{i \neq k} \cdot \exp(z^2_i \cdot z^2_k / \tau_m)}{\sum_{j=1}^{N}{1}_{i \neq j} \cdot \exp(z^2_i \cdot z^2_j / \tau_m)}
            w^2_{ik} = \lambda \cdot \mathbbm{1}_{i=k} + (1 - \lambda) \cdot s^2_{ik}
            p^1_{ik} = \frac{\exp(z^1_i \cdot z^2_k / \tau)}{\sum_{j=1}^{N}\exp(z^1_i \cdot z^2_j / \tau)}

        Args:
            q (Tensor): The representations of the queries.
            k (Tensor): The representations of the keys.

        Returns:
            Dict[str, Tensor]: The loss, logits, labels and similarities.
        """
        batch_size = q.shape[0]

        labels = torch.zeros(
            batch_size, dtype=torch.long, device=self.device)
        sim_q_ktarget = torch.einsum(
            'nc,nc->n', [q, k]).unsqueeze(-1)
        sim_k_ktarget = torch.zeros(
            batch_size, device=self.device).unsqueeze(-1)

        queue = self.queue.clone().detach()
        sim_k_queue = torch.einsum(
            'nc,ck->nk', [k, queue])
        sim_q_queue = torch.einsum(
            'nc,ck->nk', [q, queue])

        sim_k = torch.cat([sim_k_ktarget, sim_k_queue], dim=1)
        sim_q = torch.cat([sim_q_ktarget, sim_q_queue], dim=1)

        mask = nn.functional.one_hot(
            labels, 1 + queue.shape[1])

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m

        prob_k = nn.functional.softmax(logits_k, dim=1)
        prob_q = nn.functional.normalize(
            self.coeff * mask + (1 - self.coeff) * prob_k, p=1, dim=1)

        loss = - \
            torch.sum(prob_q * nn.functional.log_softmax(logits_q,
                      dim=1), dim=1).mean(dim=0)

        return loss

