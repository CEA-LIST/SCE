# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import warnings
import re

import torch


def get_matching_files_in_dir(dir: str, file_pattern: str) -> List[Path]:
    """Retrieve files in directory matching a pattern.

    Args:
        dir (str): Directory path.
        file_pattern (str): Pattern for the files.

    Raises:
        NotADirectoryError: If dir does not exist or is not a directory.

    Returns:
        List[Path]: [description]
    """
    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        files = list(dir.glob(file_pattern))
        return files
    else:
        raise NotADirectoryError(
            f'Directory "{dir}" does not exist or is not a directory')


def get_ckpts_in_dir(dir: str, ckpt_pattern: str = '*.ckpt') -> List[Path]:
    """Get all checkpoints in a directory.

    Args:
        dir (str): Directory path containing the checkpoints.
        ckpt_pattern (str): Checkpoint glob pattern. Defaults to '*.ckpt'.

    Returns:
        List[Path]: List of checkpoints paths in directory.
    """
    try:
        files = get_matching_files_in_dir(dir, ckpt_pattern)
    except NotADirectoryError:
        warnings.warn(
            f'No checkpoint found in: {dir}', category=RuntimeWarning)
        files = []
    return files


def get_last_ckpt_in_dir(
    dir: str,
    ckpt_pattern: str = '*.ckpt',
    key_sort: Callable = lambda x: x.stat().st_mtime,
) -> Optional[Path]:
    """Get last ckpt in directory following a sorting function.

    Args:
        dir (str): Directory path containing the checkpoints.
        ckpt_pattern (str, optional): Checkpoint glob pattern. Defaults to '*.ckpt'.
        key_sort (Callable, optional): Function to sort the checkpoints. Defaults to last executed time.

    Returns:
        Optional[Path]: Last checkpoint in dir, if it exists, according to key_sort.
    """
    ckpts = get_ckpts_in_dir(dir, ckpt_pattern)
    if ckpts == []:
        return None
    ckpts.sort(key=key_sort, reverse=False)

    return ckpts[-1]

def get_last_ckpt_in_path_or_dir(
    checkpoint_file: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    ckpt_pattern: str = '*.ckpt',
    key_sort: Callable = lambda x: x.stat().st_mtime,
) -> Optional[Path]:
    """Get checkpoint from file or from last checkpoint in directory following a sorting function.

    Args:
        checkpoint_file (Optional[str]): Checkpoint file path containing the checkpoint. Defaults to None.
        checkpoint_dir (Optional[str]): Directory path containing the checkpoints. Defaults to None.
        ckpt_pattern (str, optional): Checkpoint glob pattern. Defaults to '*.ckpt'.
        key_sort (Callable, optional): Function to sort the checkpoints. Defaults to last executed time.

    Returns:
        Optional[Path]: Checkpoint file if it exists or last checkpoint in dir according to key_sort.
    """
    if checkpoint_file is not None:
        checkpoint_file_path = Path(checkpoint_file)
        if checkpoint_file_path.exists() and checkpoint_file_path.is_file():
            return checkpoint_file_path
        else:
            warnings.warn(f'{checkpoint_file} is not a file or do not exist.', category=RuntimeWarning)
    if checkpoint_dir is not None:
        return get_last_ckpt_in_dir(checkpoint_dir, ckpt_pattern=ckpt_pattern, key_sort=key_sort)
    return None


def get_sub_state_dict_from_pl_ckpt(
    checkpoint_path: str,
    pattern: str = r'^(trunk\.)'
) -> Dict[Any, Any]:
    """Retrieve sub state dict from a pytorch lightning checkpoint.

    Args:
        checkpoint_path (str): Pytorch lightning checkpoint path.
        pattern (str): Pattern to filter the keys for the sub state dictionary. Defaults to r'^(trunk\.)'.

    Returns:
        Dict[Any, Any]: Sub state dict from the checkpoint following the pattern.
    """
    model = torch.load(checkpoint_path)
    state_dict = {k: v for k,
                  v in model['state_dict'].items() if re.match(pattern, k)}
    return state_dict


def remove_pattern_in_keys_from_dict(d: Dict[Any, Any], pattern: str) -> Dict[Any, Any]:
    """Remove the pattern from keys in a dictionary.

    Args:
        d (Dict[Any, Any]): The dictionary.
        pattern (str): Pattern to remove from the keys.

    Returns:
        Dict[Any, Any]: Input dictionary with updated keys. 
    """
    return {re.sub(pattern, '', k): v for k, v in d.items()}
