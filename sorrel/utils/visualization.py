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

# Import sorrel-specific packages
from sorrel.worlds import Gridworld

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #


# Decoded, resized sprite tiles are cached by (sprite path, tile size) so repeated tiles
# (walls, floors, ...) are only ever read from disk and decoded once per process.
_sprite_cache: dict[tuple[Path, tuple[int, int]], np.ndarray] = {}


def _load_tile_rgba(sprite_path: Path, tile_size: tuple[int, int]) -> np.ndarray:
    """Load a sprite as an RGBA array, resized to tile_size, caching the result."""
    key = (sprite_path, tile_size)
    tile_array = _sprite_cache.get(key)
    if tile_array is None:
        # tile_size is (row_size, col_size) i.e. (height, width), matching how it's used to
        # slice the destination array below; PIL's resize() wants (width, height), so swap.
        tile_image = (
            img.open(os.path.expanduser(sprite_path))
            .resize((tile_size[1], tile_size[0]))
            .convert("RGBA")
        )
        tile_array = np.array(tile_image, dtype=np.uint8)
        _sprite_cache[key] = tile_array
    return tile_array


def render_sprite(
    world: Gridworld,
    location: Optional[Sequence] = None,
    vision: Optional[int] = None,
    tile_size: Sequence[int] | np.ndarray = (16, 16),
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
    # If no vision or location is provided, show the whole map (do not centre on the location)
    if vision is None or location is None:
        bounds = (0, world.height - 1, 0, world.width - 1)
    else:
        vision_i = vision
        vision_j = vision
        bounds = (
            location[0] - vision_i,
            location[0] + vision_i,
            location[1] - vision_j,
            location[1] + vision_j,
        )

    tile_size = (int(tile_size[0]), int(tile_size[1]))

    # Resolved lazily, at most once per call, the first time an out-of-bounds tile is hit.
    wall_sprite: Optional[Path] = None

    # Layer handling...
    # Separate images will be generated per layer. These will be returned as a list and can then be plotted as a list.
    layers = []
    for z in range(world.map.shape[2]):

        image = np.zeros(
            (
                (bounds[1] - bounds[0] + 1) * tile_size[0],
                (bounds[3] - bounds[2] + 1) * tile_size[1],
                4,
            ),
            dtype=np.uint8,
        )

        image_i = 0
        image_j = 0

        for i in range(bounds[0], bounds[1] + 1):
            for j in range(bounds[2], bounds[3] + 1):
                if i < 0 or j < 0 or i >= world.map.shape[0] or j >= world.map.shape[1]:
                    # Tile is out of bounds, use the wall sprite
                    if wall_sprite is None:
                        wall_sprite = world.get_entities_of_kind("Wall")[0].sprite
                    tile_appearance = wall_sprite
                else:
                    tile_appearance = world.map[i, j, z].sprite

                image[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                    :,
                ] = _load_tile_rgba(tile_appearance, tile_size)

                image_j += 1
            image_i += 1
            image_j = 0

        layers.append(image)
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
    frames: Sequence[img.Image],
    filename: str,
    folder: str | os.PathLike,
) -> None:
    """Take an array of frames and assemble them into a GIF with the given path.

    Parameters:
        frames: the array of frames \n
        filename: A filename to save the images to \n
        folder: The path to save the gif to
    """
    if not frames:
        raise ValueError("animate_gif() called with no frames to save.")

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
        disposal=2,
        loop=0,
    )


class ImageRenderer:
    """Container for building images.

    Attributes:
        experiment_name (str): The name of the experiment.
        record_period (int): How often to create an animation.
        num_turns (int): The number of turns per game.
        frames (list[img.Image]): The frames to animate.
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
        self.frames: list[img.Image] = []

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
