# Import base packages
import numpy as np

# Import gem packages
from agentarium.environments import GridworldEnv
from agentarium.utils.helpers import shift


def visual_field(
    env: GridworldEnv,
    entity_map: dict[str, list[float]],
    vision: int | None = None,
    location: tuple | None = None,
    fill_entity_kind: str = "Wall",
) -> np.ndarray:
    """
    Visualize the world.

    Parameters:
        env (GridworldEnv): The environment to visualize.
        entity_map (dict[str, list[float]]): The mapping
        between objects and visual appearance.
        vision (Optional, int): The agent's visual field
        radius. If None, the entire environment. Defaults to None.
        location: (Optional, tuple): The location to centre
        the visual field on. If None, the entire environment. Defaults to None.
        fill_entity_kind (Optional, str): if the agent's vision is out of bounds,
        fill the space with appearances of this entity. Defaults to "Wall".

    Returns:
        np.ndarray: An array with dtype float64 of shape
        `(number of channels, 2 * vision + 1, 2 * vision + 1)`.
        Or if vision is None:
        `(number of channels, env.width, env.layers)`.
        Here, the number channels is determined based on the one-hot entity map provided.
    """
    # Get the number of channels used by the model.
    num_channels = len(list(entity_map.values())[0])

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack(
        [np.zeros_like(env.world, dtype=np.float64) for _ in range(num_channels)],
        axis=0,
    )

    # Iterate through the world and assign the appearance of the object at that location
    for index, x in np.ndenumerate(env.world):
        # Return visualization image
        new[:, index] = entity_map[x.kind]
    # sum the one-hot code over the layers
    new = np.sum(new, axis=-1)

    # If no location, return the full visual field
    if location is None:
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
                np.subtract(
                    [env.world.shape[0] // 2, env.world.shape[1] // 2], location[0:2]
                ),
            )
        )

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
        new = shift(array=new, shift=shift_dims, cval=np.nan)

        # Set up the dimensions of the array to crop
        crop_h = (
            env.world.shape[0] // 2 - vision,
            env.world.shape[0] // 2 + vision + 1,
        )
        crop_w = (
            env.world.shape[1] // 2 - vision,
            env.world.shape[1] // 2 + vision + 1,
        )
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index
            # If the current location was outside the shift coordinates...
            if np.isnan(x):
                # ...replace the appearance with the appearance of the fill entity in this channel.
                new[index] = entity_map[fill_entity_kind][C]
        # Return the agent's sliced observation space
        new = new.astype(np.float64)
        # ==rotate==#
        # if hasattr(env.world[location], "direction"):
        #     new = np.rot90(new, k=env.world[location].direction % 4, axes=(1, 2)).copy()
        # ==========#
        return new


def visual_field_ascii(
    env: GridworldEnv,
    entity_map: dict[str, str],
    vision: int | None = None,
    location: tuple | None = None,
    fill_entity_kind: str = "Wall",
) -> np.ndarray:
    """
    Visualize the world with ascii appearances.
    If the world has multiple layers,
    and there are multiple non-empty entities on different layers at the same horizontal coordinate,
    only the top (i.e. highest layer) non-empty entity at that coordinate will be visualized.

    Parameters:
        env (GridworldEnv): The environment to visualize.
        entity_map (dict[str, str]): The mapping
        between objects and visual appearance, where the visual appearance must be a character.
        vision (Optional, int): The agent's visual field
        radius. If None, the entire environment. Defaults to None.
        location: (Optional, tuple): The location to centre
        the visual field on. If None, the entire environment. Defaults to None.
        fill_entity_kind (Optional, str): if the agent's vision is out of bounds,
        fill the space with appearances of this entity. Defaults to "Wall".

    Returns:
        np.ndarray: An array of strings of shape
        `(2 * vision + 1, 2 * vision + 1)`.
        Or if vision is None:
        `(env.height, env.width)`.
    """

    # Create an array of equivalent shape to the world map
    new = np.empty_like(env.world, dtype=np.str_)

    # Iterate through the world and assign the appearance of the object at that location
    for index, _ in np.ndenumerate(env.world[:, :, 0]):
        H, W = index
        # iterate from top to bottom
        for L in reversed(range(env.world.shape[2])):
            # if the entity is not empty, get its appearance, and we don't need to check the lower layers.
            if env.world[H, W, L].kind != "EmptyEntity":
                new[H, W] = entity_map[env.world[H, W, L].kind]
                break
            # continue to check the lower layers if the entity is not empty.
            else:
                new[H, W] = entity_map[env.world[H, W, L].kind]

    # If no location, return the full visual field
    if location is None:
        return new.astype(np.str_)

    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.subtract(
            [env.world.shape[0] // 2, env.world.shape[1] // 2], location[0:2]
        )

        # Shift the array, and fill the appearances of coordinates outside the map with the fill entity's appearance.
        new = shift(array=new, shift=shift_dims, cval=entity_map[fill_entity_kind])

        # Set up the dimensions of the array to crop
        crop_h = (
            env.world.shape[0] // 2 - vision,
            env.world.shape[0] // 2 + vision + 1,
        )
        crop_w = (
            env.world.shape[1] // 2 - vision,
            env.world.shape[1] // 2 + vision + 1,
        )
        # Crop the array to the selected dimensions
        new = new[slice(*crop_h), slice(*crop_w)]

        # Return the agent's sliced observation space
        return new.astype(np.str_)
