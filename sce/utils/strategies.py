# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from typing import Any, List
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Before pytorch_lightning 1.6.x strategies were training type plugins. After the api changed to migrate to strategies.
if pl.__version__ < '1.6.0':
    from pytorch_lightning.plugins import TrainingTypePlugin
    from pytorch_lightning.plugins import DDPPlugin, DDP2Plugin, DDPSpawnPlugin, DeepSpeedPlugin, DataParallelPlugin, DDPFullyShardedPlugin, HorovodPlugin, DDPShardedPlugin, DDPSpawnShardedPlugin, SingleDevicePlugin, SingleTPUPlugin, TPUSpawnPlugin
    process_independent_strategies: List[TrainingTypePlugin] = [DDPPlugin, DDPSpawnPlugin, DeepSpeedPlugin,
                                                                DDPFullyShardedPlugin, HorovodPlugin, DDPShardedPlugin, DDPSpawnShardedPlugin]
    node_independent_strategies: List[TrainingTypePlugin] = [
        DDP2Plugin, DataParallelPlugin]
    fully_dependent_strategies: List[TrainingTypePlugin] = [
        SingleDevicePlugin]
    tpu_strategies: List[TrainingTypePlugin] = [
        SingleTPUPlugin, TPUSpawnPlugin
    ]

    supported_strategies: List[TrainingTypePlugin] = process_independent_strategies + \
        node_independent_strategies + fully_dependent_strategies

else:
    from pytorch_lightning.strategies import Strategy, DDPStrategy, DDP2Strategy, DDPSpawnStrategy, DeepSpeedStrategy, DataParallelStrategy, DDPFullyShardedStrategy, HorovodStrategy, DDPShardedStrategy, DDPSpawnShardedStrategy, SingleDeviceStrategy, SingleTPUStrategy, TPUSpawnStrategy

    process_independent_strategies: List[Strategy] = [DDPStrategy, DDPSpawnStrategy, DeepSpeedStrategy,
                                                      DDPFullyShardedStrategy, HorovodStrategy, DDPShardedStrategy, DDPSpawnShardedStrategy]
    node_independent_strategies: List[Strategy] = [
        DDP2Strategy, DataParallelStrategy]
    fully_dependent_strategies: List[Strategy] = [
        SingleDeviceStrategy]

    tpu_strategies: List[Strategy] = [
        SingleTPUStrategy, TPUSpawnStrategy
    ]

    supported_strategies: List[Strategy] = process_independent_strategies + \
        node_independent_strategies + fully_dependent_strategies


def get_global_batch_size_in_trainer(
    local_batch_size: int,
    trainer: Trainer,
) -> int:
    """Get global batch size used by a trainer based on the local batch size.

    Args:
        local_batch_size (int): The local_batch_size used by the trainer.
        trainer (Any): The trainer used.
    Raises:
        AttributeError: The strategy is not supported 

    Returns:
        int: The global batch size.
    """
    strategy = get_trainer_strategy(trainer)
    tpu_cores = trainer.tpu_cores
    gpus = trainer.num_gpus
    num_nodes = trainer.num_nodes
    if not any([isinstance(strategy, supported_strategy) for supported_strategy in supported_strategies]):
        raise AttributeError(f'Strategy {strategy} not supported.')
    else:
        if any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies]):
            return local_batch_size * tpu_cores
        elif any([isinstance(strategy, process_independent_strategy) for process_independent_strategy in process_independent_strategies]):
            return local_batch_size * gpus * num_nodes
        elif any([isinstance(strategy, node_independent_strategy) for node_independent_strategy in node_independent_strategies]):
            return local_batch_size * num_nodes
        elif any([isinstance(strategy, fully_dependent_strategy) for fully_dependent_strategy in fully_dependent_strategies]):
            return local_batch_size
        else:
            raise AttributeError(f'Strategy {strategy} not supported.')


def get_local_batch_size_in_trainer(
    global_batch_size: int,
    trainer: Trainer,
) -> int:
    """Get local batch size used by a trainer based on the global batch size.

    Args:
        global_batch_size (int): The global_batch_size used by the trainer.
        strategy (Any): The trainer used.
        gpus (Optional[int]): The number of gpus used by the trainer. Defaults to None.
        num_nodes (int): The number of nodes used by the trainer.
        tpu_cores (int): The number of tpu cores used by the trainer.
    Raises:
        AttributeError: The strategy is not supported 

    Returns:
        int: The local batch size.
    """
    strategy = get_trainer_strategy(trainer)
    tpu_cores = trainer.tpu_cores
    gpus = trainer.num_gpus
    num_nodes = trainer.num_nodes
    if not any([isinstance(strategy, supported_strategy) for supported_strategy in supported_strategies]):
        raise AttributeError(f'Strategy {strategy} not supported.')
    else:
        if any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies]):
            return global_batch_size // tpu_cores
        elif any([isinstance(strategy, process_independent_strategy) for process_independent_strategy in process_independent_strategies]):
            return global_batch_size // gpus // num_nodes
        elif any([isinstance(strategy, node_independent_strategy) for node_independent_strategy in node_independent_strategies]):
            return global_batch_size // num_nodes
        elif any([isinstance(strategy, fully_dependent_strategy) for fully_dependent_strategy in fully_dependent_strategies]):
            return global_batch_size
        else:
            raise AttributeError(f'Strategy {strategy} not supported.')


def get_num_devices_in_trainer(
    trainer: Trainer,
) -> int:
    """Get the number of devices used by the trainer.

    Args:
        trainer (Trainer): The trainer.

    Raises:
        AttributeError: The strategy used by trainer is not supported 

    Returns:
        int: The number of devices used by trainer.
    """
    strategy = get_trainer_strategy(trainer)
    if not any([isinstance(strategy, supported_strategy) for supported_strategy in supported_strategies]):
        raise AttributeError(f'Strategy {strategy} not supported.')
    else:
        if any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies]):
            return trainer.tpu_cores
        elif any([isinstance(strategy, process_independent_strategy) for process_independent_strategy in process_independent_strategies]):
            return trainer.num_gpus * trainer.num_nodes
        elif any([isinstance(strategy, node_independent_strategy) for node_independent_strategy in node_independent_strategies]):
            return trainer.num_gpus * trainer.num_nodes
        elif any([isinstance(strategy, fully_dependent_strategy) for fully_dependent_strategy in fully_dependent_strategies]):
            return 1
        else:
            raise AttributeError(f'Strategy {strategy} not supported.')


def get_trainer_strategy(trainer: Trainer) -> Any:
    """Retrieve the strategy from a trainer.

    Args:
        trainer (Trainer): The trainer.

    Returns:
        Any: The strategy
    """
    if pl.__version__ < '1.6.0':
        return trainer.training_type_plugin
    else:
        return trainer.strategy


def is_strategy_ddp(strategy: Any) -> bool:
    return any([isinstance(strategy, process_strategy) for process_strategy in process_independent_strategies])


def is_strategy_dp(strategy: Any) -> bool:
    return any([isinstance(strategy, node_strategy) for node_strategy in node_independent_strategies])


def is_strategy_tpu(strategy: Any) -> bool:
    return any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies])
