# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import hydra
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Any, Iterable, List, Optional, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch import nn, Tensor
from torch.nn.parameter import Parameter


class SiameseBaseModel(pl.LightningModule, ABC):
    """Abstract class to represent siamese models.

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
        num_local_crops: int = 0
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.trunk = hydra.utils.instantiate(trunk)
        self.projector = hydra.utils.instantiate(
            projector) if projector is not None else None
        self.predictor = hydra.utils.instantiate(
            predictor) if predictor is not None and self.projector is not None else None

        self.transform = hydra.utils.instantiate(
            transform) if transform is not None else None

        if transform is not None:
            rank_zero_info(f"GPU transform: {self.transform}")

        self.optimizer_cfg = optimizer

        self.normalize_outputs = normalize_outputs
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_crops = self.num_global_crops + self.num_local_crops

    @property
    def learnable_params(self) -> List[Parameter]:
        """learnable_params (List[Parameter]): List of parameters to learn composed of the trunk, and optionally the projector and predictor."""

        params = []
        params.extend(self.trunk.parameters())

        if self.projector is not None:
            params.extend(self.projector.parameters())

        if self.predictor is not None:
            params.extend(self.predictor.parameters())
        return params

    @property
    def training_steps_per_epoch(self) -> Optional[int]:
        """Total training steps inferred from datamodule and devices."""
        # TODO use pl.__version__ when the train_dataloader is initialized in trainer before configure_optimizers is called.
        # if pl.__version__ >= '1.x.0': return len(self.trainer.train_dataloader)
        if self.trainer.datamodule is not None:
            return self.trainer.datamodule.train_num_samples // self.trainer.datamodule.train_global_batch_size
        else:
            return None

    def configure_optimizers(self) -> Dict[Any, Any]:
        optimizer, scheduler = hydra.utils.instantiate(
            self.optimizer_cfg, num_steps_per_epoch=self.training_steps_per_epoch, model=self)

        if scheduler is None:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def forward(self, x: Tensor) -> Tensor:
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h
        q = self.predictor(z) if self.predictor is not None else z
        return q

    def multi_crop_shared_step(self, x: List[Tensor]) -> Dict[str, Tensor]:
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, h = 0, torch.empty(0, device=x[0].device) 
        for _, end_idx in enumerate(idx_crops):
            h_output = self.trunk(torch.cat(x[start_idx: end_idx]))
            start_idx = end_idx
            h = torch.cat((h, h_output))
        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            q = z

        return {'h': h, 'z': z, 'q': q}
        
    def shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Shared step that pass the input tensor in transforms, the trunk, projector and predictor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            q = z

        return {'h': h, 'z': z, 'q': q}

    def val_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Validation shared step that pass the input tensor in the trunk, projector and predictor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            q = z

        return {'h': h, 'z': z, 'q': q}

    def up_to_projector_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Shared step that pass the input tensor in the trunk and projector.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The computed representations.
        """
        if self.transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.transform(x)

        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h
        if self.hparams.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {'h': h, 'z': z}

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def training_step(self, batch: Iterable[Any], batch_idx: int):
        pass

    def validation_step(self, batch: Iterable[Any], batch_idx: int):
        x = batch["input"]

        return self.val_shared_step(x)
