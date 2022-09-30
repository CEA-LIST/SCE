# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

import hydra
import os
import warnings
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info

from sce.utils.checkpoints import get_last_ckpt_in_path_or_dir


@hydra.main(config_path="../sce/configs/run/pretrain/sce/resnet18", config_name="resnet18_cifar10")
def main(config: DictConfig) -> None:
    rundir = Path(to_absolute_path(config.dir.run))
    rundir.mkdir(parents=True, exist_ok=True)
    os.chdir(rundir)
    rank_zero_info(f'Run directory: {rundir}')

    hydradir = rundir / 'config/'
    hydradir.mkdir(parents=True, exist_ok=True)
    config_file = hydradir / 'pretrain.yaml'
    
    resolved_config = OmegaConf.to_yaml(config, resolve=True)

    # Save resolved config
    with config_file.open(mode='w') as f:
        f.write(resolved_config)

    # Fix seed, if seed everything: fix seed for python, numpy and pytorch
    if config.get('seed'):
        hydra.utils.instantiate(config.seed)
    else:
        warnings.warn('No seed fixed, the results are not reproducible.')

    # Create callbacks
    callbacks = []
    if config.get('callbacks'):
        for _, callback_cfg in config.callbacks.items():
            callback: Callback = hydra.utils.instantiate(callback_cfg)
            callbacks.append(callback)

    # Create trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks)

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule)

    # Create model
    model: LightningModule = hydra.utils.instantiate(
        config.model)

    # Search for last checkpoint if it exists
    model_ckpt_dirpath = config.callbacks.model_checkpoint.dirpath if config.callbacks.get(
        'model_checkpoint') else None
    ckpt_path = get_last_ckpt_in_path_or_dir(
        config.ckpt_path, model_ckpt_dirpath)
    if ckpt_path is not None:
        warnings.warn(
            f'A checkpoint has been found and loaded from this file: {ckpt_path}', category=RuntimeWarning)
    
    rank_zero_info(resolved_config)
    rank_zero_info(model)

    # Fit the trainer
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
