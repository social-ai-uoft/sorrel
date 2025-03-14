# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import numpy as np
import os
from PIL import Image

from PIL.PngImagePlugin import PngImageFile
from matplotlib import pyplot as plt
from typing import Sequence, Optional

# Import sorrel-specific packages
from sorrel.environments import GridworldEnv

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #

def visual_field_sprite(
    env: GridworldEnv,
    location: Optional[Sequence] = None,
    vision: Optional[int] = None,
    tile_size: Sequence[int] = [16, 16],
) -> list[np.ndarray]:
    """
    Create an agent visual field of size (2k + 1, 2k + 1) tiles

    Parameters:
        location: (Sequence, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        tile_size: (Sequence[int]) defines the size of the sprites. Default: 16 x 16.

    Returns:
        A list of np.ndarrays of C x H x W, determined either by the world size or the vision size.
    """
    world = env.world

    # get wall sprite
    wall_sprite = env.get_entities_of_kind("Wall")[0].sprite

    # If no vision or location is provided, show the whole map (do not centre on the location)
    if vision is None or location is None:
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
    filename: str | os.PathLike,
    folder: str | os.PathLike,
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