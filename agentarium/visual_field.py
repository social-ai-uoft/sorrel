# Import base packages
import numpy as np
import jax

from typing import Optional
from numpy.typing import ArrayLike

# Import gem packages
from agentarium.primitives import GridworldEnv
from agentarium.utils import shift

# TODO: color_map isn't always required, so why does it not default to None?
def visual_field(
    world: np.ndarray,
    color_map,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    channels: int = 5,
    return_rgb=False,
) -> np.ndarray:
    """
    Visualize the world.

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        channels: (int) defines the size of the visualization. By default, 5 channels. \n
        return_rgb: (bool) Whether to return the image as a plottable RGB image.

    Returns:
        An np.ndarray of C x H x W, determined either by the world size or the vision size.
    """
    C = channels  # Number of channels
    if return_rgb:
        C = 3
        colors = color_map(C)

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack([np.zeros_like(world, dtype=np.float64) for _ in range(C)], axis=0)[
        :, :, :, 0
    ]
    # Get wall appearance from the world object (just pick the first wall object for simplicity)
    wall_appearance = GridworldEnv.get_entities_(world, "Wall")[0].appearance

    # Iterate through the world and assign the appearance of the object at that location
    for index, _ in np.ndenumerate(world[:, :, 0]):
        H, W = index  # Get the coordinates
        # Return visualization image
        if return_rgb:
            new[:, H, W] = colors[world[H, W, 0].kind]
        else:
            # print(world[H, W, 0], world[H, W, 0].appearance)
            new[:, H, W] = world[H, W, 0].appearance

    # If no location, return the full visual field
    if location is None:
        if return_rgb:
            return new.astype(np.uint8).transpose(
                (1, 2, 0)
            )  # Pyplot wants inputs organized in H x W x C
        else:
            return new.astype(np.float64)

    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.hstack(
            (
                [
                    0
                ],  # The first dimension is zero, because the channels are not shifted
                np.subtract([world.shape[0] // 2, world.shape[1] // 2], location[0:2]),
            )
        )

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
        new = shift(array=new, shift=shift_dims, cval=np.nan)

        # Set up the dimensions of the array to crop
        crop_h = (world.shape[0] // 2 - vision, world.shape[0] // 2 + vision + 1)
        crop_w = (world.shape[1] // 2 - vision, world.shape[1] // 2 + vision + 1)
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index  # Get the coordinates
            if np.isnan(
                x
            ):  # If the current location was outside the shift coordinates...
                # ...replace the appearance with the wall appearance in this channel.
                if return_rgb:
                    new[index] = colors["Wall"][C]
                else:
                    new[index] = wall_appearance[C]

        # Return the agent's sliced observation space
        k = (
            0
            if not hasattr(world[location], "direction")
            else world[location].direction % 4
        )

        if return_rgb:
            new = new.astype(np.uint8).transpose((1, 2, 0))
            # ==rotate==#
            new = jax.numpy.rot90(new, k=k)
            # ==========#
            return new
        else:
            new = new.astype(np.float64)
            # ==rotate==#
            new = np.rot90(new, k=k, axes=(1, 2)).copy()
            # ==========#
            return new


def visual_field_multilayer(
    world: np.ndarray,
    color_map,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    channels: int = 5,
    return_rgb=False,
) -> np.ndarray:
    """
    Visualize the world.

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        channels: (int) defines the size of the visualization. By default, 5 channels. \n
        return_rgb: (bool) Whether to return the image as a plottable RGB image.

    Returns:
        An np.ndarray of C x H x W, determined either by the world size or the vision size.
    """
    C = channels  # Number of channels
    if return_rgb:
        C = 3
        colors = color_map(C)

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack(
        [np.zeros_like(world, dtype=np.float64) for _ in range(C)], axis=0
    ).squeeze()
    # Get wall appearance from the world object (just pick the first wall object for simplicity)
    wall_appearance = GridworldEnv.get_entities_(world, "Wall")[0].appearance

    # Iterate through the world and assign the appearance of the object at that location
    for layer in range(world.shape[-1]):
        for index, _ in np.ndenumerate(world[:, :, layer]):
            H, W = index  # Get the coordinates
            # Return visualization image
            if return_rgb:
                if world.shape[-1] > 1:
                    new[:, H, W, layer] = colors[world[H, W, layer].kind]
                else:
                    new[:, H, W] = colors[world[H, W, layer].kind]
            else:
                if world.shape[-1] > 1:
                    new[:, H, W, layer] = world[H, W, layer].appearance
                else:
                    new[:, H, W] = world[H, W, layer].appearance
    else:
        new = np.sum(new, axis=-1)

    # If no location, return the full visual field
    if location is None:
        if return_rgb:
            return new.astype(np.uint8).transpose(
                (1, 2, 0)
            )  # Pyplot wants inputs organized in H x W x C
        else:
            return new.astype(np.float64)

    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.hstack(
            (
                [
                    0
                ],  # The first dimension is zero, because the channels are not shifted
                np.subtract([world.shape[0] // 2, world.shape[1] // 2], location[0:2]),
            )
        )

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
        new = shift(array=new, shift=shift_dims, cval=np.nan)

        # Set up the dimensions of the array to crop
        crop_h = (world.shape[0] // 2 - vision, world.shape[0] // 2 + vision + 1)
        crop_w = (world.shape[1] // 2 - vision, world.shape[1] // 2 + vision + 1)
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index  # Get the coordinates
            if np.isnan(
                x
            ):  # If the current location was outside the shift coordinates...
                # ...replace the appearance with the wall appearance in this channel.
                if return_rgb:
                    new[index] = colors["Wall"][C]
                else:
                    new[index] = wall_appearance[C]
        # Return the agent's sliced observation space
        if return_rgb:
            new = new.astype(np.uint8).transpose((1, 2, 0))
            # ==rotate==#
            new = jax.numpy.rot90(new, k=world[location].direction % 4)
            # ==========#
            return new
        else:
            new = new.astype(np.float64)
            # ==rotate==#
            new = np.rot90(new, k=world[location].direction % 4, axes=(1, 2)).copy()
            # ==========#
            return new
