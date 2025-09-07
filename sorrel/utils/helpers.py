# --------------- #
# region: Imports #
# --------------- #

import random

# Import base packages
from typing import Any, Sequence

import numpy as np
import torch

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Helper functions    #
# --------------------------- #


def set_seed(seed: int) -> None:
    r"""Sets a seed for replication. Sets the seed for :meth:`random.seed()`,
    :meth:`numpy.random.seed()`, and :meth:`torch.manual_seed()`.

    Args:
        seed: An int setting the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_seed() -> int:
    r"""Generates a random seed.

    Returns:
        The value of the seed generated
    """
    seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def shift(
    array: np.ndarray, shift: Sequence | np.ndarray, cval: Any = np.nan
) -> np.ndarray:
    r"""Returns copy of array shifted by offset, with fill using constant.

    Args:
        array: The array to shift.
        shift: A sequence of dimensions equivalent to the array passed into the function.
        cval: The value to replace any new elements introduced into the offset array. By default, replaces them with nan's.

    Returns:
        np.ndarray: The shifted array.
    """
    offset = np.atleast_1d(shift)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = array[
        tuple(slice1(-o) for o in offset)
    ]

    for axis, o in enumerate(offset):
        new_array[
            (slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)
        ] = cval

    return new_array


def nearest_2_power(n: int) -> int:
    r"""Computes the next power of 2.

    Useful for programmatically shifting batch and buffer sizes to computationally
    efficient values.

    Args:
        n: The number.

    Returns:
        int: The nearest power of two equal to or larger than n.
    """

    # Bit shift counter
    bit_shifts = 0

    # If `n` is already a power of 2 (bitwise n & (n - 1)),
    # return `n` (unless n is 0, handled below)
    if n and not (n & (n - 1)):
        return n

    # Otherwise, repeatedly shift `n` rightwards by 1 bit
    # until `n` is 0...
    while n != 0:
        n >>= 1
        bit_shifts += 1

    # ...then left shift 1 by the number of times n was shifted
    return 1 << bit_shifts


def clip(n: int, minimum: int, maximum: int) -> int:
    r"""Clips an input to a number between the minimum and maximum values passed into the
    function.

    Args:
        n: The number.
        minimum: The minimum number.
        maximum: The maximum number.

    Returns:
        int: The clipped value of n.
    """
    if n < minimum:
        return minimum
    elif n > maximum:
        return maximum
    else:
        return n


def one_hot_encode(value: int, num_classes: int) -> np.ndarray:
    r"""Create a numpy array of shape (num_classes, ) that encodes the position of
    `value` as a one-hot vector.

    Args:
        value: The position to one-hot encode.
        num_classes: The length of the one-hot vector.

    .. note:: `value` cannot be larger than `num_classes - 1` without leading to an index error.
    """
    assert value <= (
        num_classes - 1
    ), f"The maximum value of `value` is {num_classes - 1}."

    # Create a zero array of length num_classes
    one_hot = np.zeros(num_classes)

    # Set the index corresponding to 'value' to 1
    one_hot[value] = 1

    return one_hot


# --------------------------- #
# endregion                   #
# --------------------------- #
