# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

def warmup_value(initial_value: float, final_value: float, step: int = 0, max_step: int = 0):
    """
    Apply warmup to a value.

    Args:
        initial_value (float): Initial value.
        final_value (float): Final value.
        step (int, optional): Current step. Defaults to 0.
        max_step (int, optional): Max step for warming up. Defaults to 0.

    Returns:
        float: Current value.
    """
    if step >= max_step:
        return final_value
    else:
        return initial_value + step * \
            (final_value - initial_value) / (max_step)
