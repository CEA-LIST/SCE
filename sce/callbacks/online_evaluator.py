# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

# ------------------------------------------------------------------------
# Modified from lightning-bolts (https://github.com/Lightning-AI/lightning-bolts)
# Licensed under the Apache License, Version 2.0
# -----------------------------

import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Sequence

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from torch.nn.parameter import Parameter
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional import accuracy

from sce.utils.strategies import is_strategy_ddp, is_strategy_dp


class OnlineEvaluator(Callback):
    """Attaches a classifier to evaluate a specific representation from the model during training.

    Args:
        optimizer (DictConfig): Config to instantiate an optimizer and optionnaly a scheduler.
        classifier (DictConfig): Config to intantiate a classifier.
        input_name (str): Name of the representation to evluate from the model outputs. Defaults to h.
        precision (int): Precision for the classifier that must match the model, if `16` use average mixed precision. Defaults to 32. 

    Example::

        optimizer = {...} # config to build an optimizer
        classifier = {...} # config to build a classifier
        trainer = Trainer(callbacks=[OnlineEvaluator(optimizer, classifier)])
        )
    """

    def __init__(
        self,
        optimizer: DictConfig,
        classifier: DictConfig,
        input_name: str = 'h',
        precision: int = 32
    ):
        super().__init__()

        self.input_name = input_name
        self.classifier = hydra.utils.instantiate(classifier)
        self.optimizer, self.scheduler = hydra.utils.instantiate(
            optimizer, model=self.classifier)
        self.precision = precision

        assert precision in [16, 32]

        self.use_amp = self.precision == 16
        self.scaler = GradScaler(enabled=self.use_amp)
        self._recovered_callback_state = None

    @property
    def learnable_params(self) -> List[Parameter]:
        params = list(self.classifier.parameters())
        return params

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.classifier = self.classifier.to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if is_strategy_ddp(accel.strategy):
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.classifier = DDP(
                    self.classifier, device_ids=[pl_module.device])
            elif is_strategy_dp(accel.strategy):
                from torch.nn.parallel import DataParallel as DP

                self.classifier = DP(
                    self.classifier, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        if self._recovered_callback_state is not None:
            self.classifier.load_state_dict(
                self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(
                self._recovered_callback_state["optimizer_state"])
            self.scaler.load_state_dict(
                self._recovered_callback_state["scaler"])

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:

        targets = batch["label"]

        representations = outputs[self.input_name].clone().detach()

        mask = targets != -1

        with autocast(enabled=self.use_amp):
            logits = self.classifier(representations[mask])
            loss = nn.functional.cross_entropy(logits, targets[mask])

        online_acc_1 = accuracy(logits, targets)
        online_acc_5 = accuracy(logits, targets, top_k=5)

        # update finetune weights
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.scheduler != None:
            self.scheduler.step()

        pl_module.log('online/train_acc_1', online_acc_1, on_step=True,
                      on_epoch=True, prog_bar=True)
        pl_module.log('online/train_acc_5', online_acc_5,
                      on_step=True, on_epoch=True)
        pl_module.log('online/train_loss', loss, on_step=True, on_epoch=True)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.classifier.eval()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.classifier.train()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int
    ):
        targets = batch["label"]

        representations = outputs[self.input_name].clone()

        mask = targets != -1

        with autocast(enabled=self.use_amp):
            logits = self.classifier(representations[mask])
            loss = nn.functional.cross_entropy(logits, targets[mask])

        val_acc_1 = accuracy(logits, targets)
        val_acc_5 = accuracy(logits, targets, top_k=5)

        pl_module.log('online/val_acc_1', val_acc_1, on_step=False,
                      on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log('online/val_acc_5', val_acc_5,
                      on_step=False, on_epoch=True)
        pl_module.log('online/val_loss', loss, on_step=False,
                      on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {"state_dict": self.classifier.state_dict(), "optimizer_state": self.optimizer.state_dict(), "scaler": self.scaler.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state
