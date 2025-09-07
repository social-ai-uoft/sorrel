# --------------- #
# region: Imports #
# --------------- #

import os
from pathlib import Path
from typing import Optional, Sequence

# Import base packages
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as img  # this is the module
from PIL.PngImagePlugin import PngImageFile

# Import sorrel-specific packages
from sorrel.worlds import Gridworld

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #


def render_sprite(
    world: Gridworld,
    location: Optional[Sequence] = None,
    vision: Optional[int] = None,
    tile_size: list[int] | np.ndarray = [16, 16],
) -> list[np.ndarray]:
    """Render a sprite of (2k + 1, 2k + 1) tiles centered at location, where k=vision.

    If vision or location is None, render the entire world.map.

    Args:
        location: defines the location to centre the visualization on. Defaults to None.
        vision: defines the size of the visualization of (2v + 1, 2v + 1) pixels. Defaults to None.
        tile_size: defines the size of the sprites. Default: 16 x 16.

    Returns:
        A list of np.ndarrays of C x H x W, determined either by the world size or the vision size.
    """

    # get wall sprite
    wall_sprite = world.get_entities_of_kind("Wall")[0].sprite

    # If no vision or location is provided, show the whole map (do not centre on the location)
    if vision is None or location is None:
        location = (world.map.shape[0] // 2, world.map.shape[1] // 2)
        # Use the largest location dimension to ensure that the entire map is visible in the event of a non-square map
        vision_i = location[0]
        vision_j = location[1]
    else:
        vision_i = vision
        vision_j = vision

    # Layer handling...
    # Separate images will be generated per layer. These will be returned as a list and can then be plotted as a list.
    layers = []
    for z in range(world.map.shape[2]):

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
                if i < 0 or j < 0 or i >= world.map.shape[0] or j >= world.map.shape[1]:
                    # Tile is out of bounds, use wall_app
                    tile_image = (
                        img.open(os.path.expanduser(wall_sprite))
                        .resize(tile_size)
                        .convert("RGBA")
                    )
                else:
                    tile_appearance = world.map[i, j, z].sprite
                    tile_image = (
                        img.open(os.path.expanduser(tile_appearance))
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
    r"""Plot helper function that takes an image or list of layers and plots it in
    Matplotlib.

    Args:
        image: A numpy array or list of numpy arrays with the image layer(s).
    """
    if isinstance(image, np.ndarray):
        plt.imshow(image)
        plt.show()
    else:
        for layer in image:
            plt.imshow(layer)
        plt.show()


def image_from_array(image: np.ndarray | list[np.ndarray]) -> img.Image:
    r"""Create a PIL image from an single-layer image or list of layers.

    Args:
        image: A numpy array or list of numpy arrays with the image layer(s).

    Returns:
        Image: A PIL image version of the image.
    """
    if isinstance(image, np.ndarray):
        output = img.fromarray(image, mode="RGBA")
    else:
        output = img.fromarray(image[0], mode="RGBA")
        for layer in image[1:]:
            next_layer = img.fromarray(layer, mode="RGBA")
            output.paste(next_layer, (0, 0), mask=next_layer)
    return output


def image_from_figure(fig) -> img.Image:
    r"""Convert a Matplotlib figure to a PIL Image.

    .. note:: DO NOT use this with plt.show(), as it will not work and will return a blank image.

    Parameters:
        fig: If in fig, axis format, then fig. If in plt format, then plt.

    Returns:
        img: An image file in PIL format.
    """
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = img.open(buf)
    return image


def animate_gif(
    frames: Sequence[PngImageFile],
    filename: str,
    folder: str | os.PathLike,
) -> None:
    """Take an array of frames and assemble them into a GIF with the given path.

    Parameters:
        frames: the array of frames \n
        filename: A filename to save the images to \n
        folder: The path to save the gif to
    """
    if not os.path.exists(folder):
        print(f"Directory {folder} does not exist; creating directory.")
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = os.path.join(folder, filename + ".gif")

    frames[0].save(
        os.path.expanduser(path),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )


class ImageRenderer:
    """Container for building images.

    Attributes:
        experiment_name (str): The name of the experiment.
        record_period (int): How often to create an animation.
        num_turns (int): The number of turns per game.
        frames (list[PngImageFile]): The frames to animate.
    """

    def __init__(self, experiment_name: str, record_period: int, num_turns: int):
        """Initialize an ImageRenderer.

        Args:
            experiment_name (str): The name of the experiment.
            record_period (int): How often to create an animation.
            num_turns (int): The number of turns per game.
        """
        self.experiment_name = experiment_name
        self.record_period = record_period
        self.num_turns = num_turns
        self.frames = []

    def clear(self):
        """Zero out the frames."""
        del self.frames[:]

    def add_image(self, world: Gridworld) -> None:
        """Add an image to the frames.

        Args:
            env (Gridworld): The environment.
            epoch (int): The epoch.
        """
        full_sprite = render_sprite(world)
        self.frames.append(image_from_array(full_sprite))

    def save_gif(self, epoch: int, folder: os.PathLike) -> None:
        """Save a gif to disk.

        Args:
            epoch (int): The epoch.
            folder (os.PathLike): The destination folder.
        """
        animate_gif(self.frames, f"{self.experiment_name}_epoch{epoch}", folder)
        # Clear frames
        self.clear()


# --------------------------- #
# endregion: Visualizations   #
# --------------------------- #
