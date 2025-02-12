# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import os
import torch
import random
import numpy as np

from typing import Optional, Union, Sequence
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

# Import gem packages
from agentarium.primitives import GridworldEnv

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #

def visual_field_sprite(
    world: np.ndarray,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    tile_size: Sequence[int] = [16, 16],
) -> list[np.ndarray]:
    """
    Create an agent visual field of size (2k + 1, 2k + 1) tiles

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        tile_size: (Sequence[int]) defines the size of the sprites. Default: 16 x 16.

    Returns:
        A list of np.ndarrays of C x H x W, determined either by the world size or the vision size.
    """

    # get wall sprite
    wall_sprite = GridworldEnv.get_entities_(world, "Wall")[0].sprite

    # If no location is provided, place the location on the centre of the map with enough space to see the whole world map
    if location is None:
        location = (world.shape[0] // 2, world.shape[1] // 2)
        # Use the largest location dimension to ensure that the entire map is visible in the event of a non-square map
        vision_i = location[0]
        vision_j = location[1]
    else:
        vision_i = vision
        vision_j = vision

    # Layer handling...
    # Separate images will be generated per layer. These will be returned as a list and can then be plotted as a list.
    layers = []
    for z in range(world.shape[2]):

        bounds = (
            location[0] - vision_i,
            location[0] + vision_i,
            location[1] - vision_j,
            location[1] + vision_j,
        )

        image_r = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_g = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_b = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_a = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )

        image_i = 0
        image_j = 0

        for i in range(bounds[0], bounds[1] + 1):
            for j in range(bounds[2], bounds[3] + 1):
                if i < 0 or j < 0 or i >= world.shape[0] or j >= world.shape[1]:
                    # Tile is out of bounds, use wall_app
                    tile_image = (
                        Image.open(os.path.expanduser(wall_sprite))
                        .resize(tile_size)
                        .convert("RGBA")
                    )
                else:
                    tile_appearance = world[i, j, z].sprite
                    tile_image = (
                        Image.open(os.path.expanduser(tile_appearance))
                        .resize(tile_size)
                        .convert("RGBA")
                    )

                tile_image_array = np.array(tile_image)
                alpha = tile_image_array[:, :, 3]
                # tile_image_array[alpha == 0, :3] = 0
                image_r[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 0]
                image_g[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 1]
                image_b[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 2]
                image_a[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 3]

                image_j += 1
            image_i += 1
            image_j = 0

        # image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        image = np.zeros((image_r.shape[0], image_r.shape[1], 4))
        image[:, :, 0] = image_r
        image[:, :, 1] = image_g
        image[:, :, 2] = image_b
        image[:, :, 3] = image_a
        layers.append(np.asarray(image, dtype=np.uint8))
    return layers


def visual_field_sprite_empty(
    world: np.ndarray,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    tile_size: Sequence[int] = [16, 16],
) -> list[np.ndarray]:
    """
    Create an agent visual field of size (2k + 1, 2k + 1) tiles

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        tile_size: (Sequence[int]) defines the size of the sprites. Default: 16 x 16.

    Returns:
        A list of np.ndarrays of C x H x W, determined either by the world size or the vision size.
    """

    # get wall sprite
    wall_sprite = GridworldEnv.get_entities_(world, "Wall")[0].sprite

    # If no location is provided, place the location on the centre of the map with enough space to see the whole world map
    if location is None:
        location = (world.shape[0] // 2, world.shape[1] // 2)
        # Use the largest location dimension to ensure that the entire map is visible in the event of a non-square map
        vision_i = location[0]
        vision_j = location[1]
    else:
        vision_i = vision
        vision_j = vision

    # Layer handling...
    # Separate images will be generated per layer. These will be returned as a list and can then be plotted as a list.
    layers = []
    for z in range(world.shape[2]):

        bounds = (
            location[0] - vision_i,
            location[0] + vision_i,
            location[1] - vision_j,
            location[1] + vision_j,
        )

        image_r = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_g = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_b = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_a = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )

        image_i = 0
        image_j = 0

        for i in range(bounds[0], bounds[1] + 1):
            for j in range(bounds[2], bounds[3] + 1):
                if i < 0 or j < 0 or i >= world.shape[0] or j >= world.shape[1]:
                    # Tile is out of bounds, use wall_app
                    tile_image = (
                        Image.open(os.path.expanduser(wall_sprite))
                        .resize(tile_size)
                        .convert("RGBA")
                    )

                    tile_image_array = np.array(tile_image)
                    alpha = tile_image_array[:, :, 3]
                    # tile_image_array[alpha == 0, :3] = 0
                    image_r[
                        image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                        image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                    ] = tile_image_array[:, :, 0]
                    image_g[
                        image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                        image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                    ] = tile_image_array[:, :, 1]
                    image_b[
                        image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                        image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                    ] = tile_image_array[:, :, 2]
                    image_a[
                        image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                        image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                    ] = tile_image_array[:, :, 3]

                    image_j += 1
            image_i += 1
            image_j = 0

        # image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        image = np.zeros((image_r.shape[0], image_r.shape[1], 4))
        image[:, :, 0] = image_r
        image[:, :, 1] = image_g
        image[:, :, 2] = image_b
        image[:, :, 3] = image_a
        layers.append(np.asarray(image, dtype=np.uint8))
    return layers


def composite_visualization(
    envs: Sequence[np.ndarray],
    tile_size: Sequence[int] = [16, 16]
):
    """Plot a composite of the world visualizations"""
    vis_worlds = []
    for count, env in enumerate(envs):
        vis = visual_field_sprite(env.world, tile_size=env.tile_size)
        vis_worlds.append(vis[0])
    template = np.zeros(vis_worlds[0].shape, dtype=np.uint8)
    for _ in range(6-len(envs)):
        vis_worlds.append(np.zeros(vis_worlds[0].shape, dtype=np.uint8))
    grid_image_h1 = np.hstack([vis_worlds[0], vis_worlds[1], vis_worlds[2]])
    grid_image_h2 = np.hstack([vis_worlds[3], vis_worlds[4], vis_worlds[5]])
    grid_image = np.vstack([grid_image_h1, grid_image_h2])
    return grid_image

    #TODO: test this func, enable row and col num specification

def plot(image: np.ndarray | list[np.ndarray]) -> None:
    """Plot helper function that takes an image or list of layers."""
    if isinstance(image, np.ndarray):
        plt.imshow(image)
        plt.show()
    else:
        for layer in image:
            plt.imshow(layer)
        plt.show()


def image_from_array(image: np.ndarray | list[np.ndarray]) -> Image:
    """Create a PIL image from an single-layer image or list of layers."""
    if isinstance(image, np.ndarray):
        output = Image.fromarray(image, mode="RGBA")
    else:
        output = Image.fromarray(image[0], mode="RGBA")
        for layer in image[1:]:
            next_layer = Image.fromarray(layer, mode="RGBA")
            output.paste(next_layer, (0, 0), mask=next_layer)
    return output


def fig2img(fig) -> Image:
    """Convert a Matplotlib figure to a PIL Image.

    NOTE: DO NOT use this with plt.show(), as it will not work and will return a blank image.

    Parameters:
        fig: If in fig, axis format, then fig. If in plt format, then plt.

    Returns:
        img: An image file in PIL format."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def animate(
    frames: Sequence[PngImageFile],
    filename: Union[str, os.PathLike],
    folder: Union[str, os.PathLike],
) -> None:
    """
    Take an array of frames and assemble them into a GIF with the given path.

    Parameters:
        frames: the array of frames \n
        filename: A filename to save the images to \n
        folder: The path to save the gif to
    """
    path = folder + filename + ".gif"

    frames[0].save(
        os.path.expanduser(path),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )


# --------------------------- #
# endregion: Visualizations   #
# --------------------------- #

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


def shift(array: ArrayLike, shift: Sequence, cval=np.nan) -> np.ndarray:
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
