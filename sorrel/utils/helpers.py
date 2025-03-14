# --------------- #
# region: Imports #
# --------------- #

# Import base packages
from typing import Sequence
import numpy as np
import torch
import random

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Helper functions    #
# --------------------------- #

def set_seed(seed: int) -> None:
    """
    Sets a seed for replication.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_seed() -> int:
    """
    Generates a random seed

    Returns:
        The value of the seed generated
    """
    seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def shift(array: Sequence, shift: Sequence, cval=np.nan) -> np.ndarray:
    """
    Returns copy of array shifted by offset, with fill using constant.

    Parameters:
        array: The array to shift. \n
        shift: A sequence of dimensions equivalent to the array passed
        into the function. \n
        cval: The value to replace any new elements introduced into the
        offset array. By default, replaces them with nan's.

    Return:
        np.ndarray: The shifted array
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
    """
    Computes the next power of 2. Useful for programmatically
    shifting batch and buffer sizes to computationally efficient
    values.
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


def minmax(n: int, minimum: int, maximum: int) -> int:
    """
    Clips an input to a number between the minimum
    and maximum values passed into the function.
    """
    if n < minimum:
        return minimum
    elif n > maximum:
        return maximum
    else:
        return n


def one_hot_encode(value, num_classes) -> np.ndarray:
    # Create a zero array of length num_classes
    one_hot = np.zeros(num_classes)

    # Set the index corresponding to 'value' to 1
    one_hot[value] = 1

    return one_hot

# --------------------------- #
# endregion                   #
# --------------------------- #