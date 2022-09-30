# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torchmetrics.classification.accuracy import Accuracy

from sce.utils.checkpoints import get_sub_state_dict_from_pl_ckpt, remove_pattern_in_keys_from_dict


class LinearClassifierEvaluation(pl.LightningModule):
    """Linear classifier evaluation for self-supervised learning.

    Args:
        trunk (DictConfig): Config tu build a trunk.
        classifier (DictConfig): Config to build a classifier.
        optimizer (DictConfig): Config to build an optimizer.
        pretrained_trunk_path (str): Path to the pretrained trunk file.
        trunk_pattern (str): Pattern to retrieve the trunk model in checkpoint state_dict and delete the key. Defaults to ^(trunk\.).

    Example::

        trunk = {...} # config to build a trunk
        classifier = {...} # config to build a classifier
        optimizer = {...} # config to build an optimizer
        pretrained_trunk_path = ... # path where the trunk has been saved

        model = LinearClassifierEvaluation(trunk, classifier, optimizer, pretrained_trunk_path)
    """

    def __init__(
        self,
        trunk: DictConfig,
        classifier: DictConfig,
        optimizer: DictConfig,
        pretrained_trunk_path: str,
        trunk_pattern: str = r'^(trunk\.)',
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_cfg = optimizer

        trunk_state_dict = get_sub_state_dict_from_pl_ckpt(
            checkpoint_path=pretrained_trunk_path, pattern=trunk_pattern)
        trunk_state_dict = remove_pattern_in_keys_from_dict(
            d=trunk_state_dict, pattern=trunk_pattern)

        self.trunk = hydra.utils.instantiate(trunk)
        self.trunk.load_state_dict(trunk_state_dict)

        for param in self.trunk.parameters():
            param.requires_grad = False

        self.classifier = hydra.utils.instantiate(classifier)

        self.train_acc_1 = Accuracy(top_k=1)
        self.train_acc_5 = Accuracy(top_k=5)
        self.val_acc_1 = Accuracy(top_k=1, compute_on_step=False)
        self.val_acc_5 = Accuracy(top_k=5, compute_on_step=False)
        self.test_acc_1 = Accuracy(top_k=1, compute_on_step=False)
        self.test_acc_5 = Accuracy(top_k=5, compute_on_step=False)

    @property
    def learnable_params(self) -> List[Parameter]:
        params = list(self.classifier.parameters())
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

    def forward(self, x: Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            h = self.trunk(x)
        preds = self.classifier(h)
        return {"preds": preds, "h": h}

    def configure_optimizers(self) -> Dict[Any, Any]:
        optimizer, scheduler = hydra.utils.instantiate(
            self.optimizer_cfg, num_steps_per_epoch=self.training_steps_per_epoch, model=self)

        if scheduler is None:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def shared_step(self, x: Tensor):
        with torch.no_grad():
            h = self.trunk(x)

        preds = self.classifier(h)

        return preds

    def on_train_epoch_start(self) -> None:
        self.trunk.eval()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, targets = batch["input"], batch["label"]

        preds = self.shared_step(x)
        loss = nn.functional.cross_entropy(preds, targets)
        preds = preds.softmax(-1)

        acc_1 = self.train_acc_1(preds, targets)
        acc_5 = self.train_acc_5(preds, targets)

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc_1", acc_1, on_epoch=True, prog_bar=True)
        self.log("train/acc_5", acc_5, on_epoch=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, targets = batch["input"], batch["label"]
        preds = self.shared_step(x)
        preds = preds.softmax(-1)
    
        loss = nn.functional.cross_entropy(preds, targets)

        self.val_acc_1(preds, targets)
        self.val_acc_5(preds, targets)

        self.log("val/loss", loss)
        self.log("val/acc_1", self.val_acc_1, prog_bar=True)
        self.log("val/acc_5", self.val_acc_5)

        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, targets = batch["input"], batch["label"]
        preds = self.shared_step(x)
        preds = preds.softmax(-1)
    
        loss = nn.functional.cross_entropy(preds, targets)

        self.test_acc_1(preds, targets)
        self.test_acc_5(preds, targets)

        self.log("test/loss", loss)
        self.log("test/acc_1", self.test_acc_1, prog_bar=True)
        self.log("test/acc_5", self.test_acc_5)

        return loss
