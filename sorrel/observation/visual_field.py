# Import base packages
import numpy as np

# sorrel imports
from sorrel.utils.helpers import shift
from sorrel.worlds import Gridworld


def visual_field(
    world: Gridworld,
    entity_map: dict[str, np.ndarray],
    vision: int | None = None,
    location: tuple | None = None,
    fill_entity_kind: str = "Wall",
) -> np.ndarray:
    """Visualize the world.

    See :py:meth:`.OneHotObservationSpec.observe()` for an example of how this function is used.

    Args:
        world: The world tovisualize.
        entity_map: The mapping between objects and visual appearance.
        vision: The agent's visual field radius.
            If None, the entire environment. Defaults to None.
        location: The location to center the visual field on.
            If None, the entire environment. Defaults to None.
        fill_entity_kind: if the agent's vision is out of bounds,
            fill the space with appearances of this entity. Defaults to "Wall".

    Returns:
        An array with dtype float64 of shape
        `(number of channels, 2 * vision + 1, 2 * vision + 1)`.
        Or if vision or location is None:
        `(number of channels, world.width, world.layers)`.
        Here, the number channels is determined based on the one-hot entity map provided.
    """
    # Get the number of channels used by the model.
    num_channels = len(list(entity_map.values())[0])

    # If no location, return the full visual field
    if location is None or vision is None:
        # Create an array of equivalent shape to the world map, with C appearance channels
        new = np.stack(
            [np.zeros_like(world.map, dtype=np.float64) for _ in range(num_channels)],
            axis=0,
        )

        # Iterate through the world and assign the appearance of the object at that location
        for index, x in np.ndenumerate(world.map):
            # Return visualization image
            new[:, *index] = entity_map[x.kind]
        # sum the one-hot code over the layers
        new = np.sum(new, axis=-1)
        return new.astype(np.float64)

    # Otherwise...
    else:
        # Only process the region of the map that the window actually covers,
        # instead of building and shifting a full-size array.
        loc_h, loc_w = location[0], location[1]
        map_h, map_w, _ = world.map.shape
        h_lo, h_hi = max(0, loc_h - vision), min(map_h, loc_h + vision + 1)
        w_lo, w_hi = max(0, loc_w - vision), min(map_w, loc_w + vision + 1)

        sub_map = world.map[h_lo:h_hi, w_lo:w_hi]
        sub = np.stack(
            [np.zeros_like(sub_map, dtype=np.float64) for _ in range(num_channels)],
            axis=0,
        )
        for index, x in np.ndenumerate(sub_map):
            sub[:, *index] = entity_map[x.kind]
        sub = np.sum(sub, axis=-1)

        window = 2 * vision + 1
        new = np.empty((num_channels, window, window), dtype=np.float64)
        new[...] = np.asarray(entity_map[fill_entity_kind], dtype=np.float64)[
            :, None, None
        ]
        dest_h, dest_w = h_lo - (loc_h - vision), w_lo - (loc_w - vision)
        new[:, dest_h : dest_h + (h_hi - h_lo), dest_w : dest_w + (w_hi - w_lo)] = sub

        # TODO: support per-agent rotation of the observation window based on
        # `world.map[location].direction`, as previously sketched here.
        return new


def visual_field_ascii(
    world: Gridworld,
    entity_map: dict[str, str],
    vision: int | None = None,
    location: tuple | None = None,
    fill_entity_kind: str = "Wall",
) -> np.ndarray:
    """Visualize the world with ascii appearances.

    If the world has multiple layers,
    and there are multiple non-empty entities on different layers at the same horizontal coordinate,
    only the top (i.e. highest layer) non-empty entity at that coordinate will be visualized.

    See :py:meth:`.AsciiObservationSpec.observe()` for an example of how this function is used.

    Args:
        world: The world tovisualize.
        entity_map: The mapping
        between objects and visual appearance, where the visual appearance must be a character.
        vision: The agent's visual field radius.
            If None, the entire environment. Defaults to None.
        location: The location to center the visual field on.
            If None, the entire environment. Defaults to None.
        fill_entity_kind: if the agent's vision is out of bounds,
            fill the space with appearances of this entity. Defaults to "Wall".

    Returns:
        An array of strings of shape
        `(2 * vision + 1, 2 * vision + 1)`.
        Or if vision or location is None:
        `(world.height, world.width)`.
    """
    # Create an array of equivalent shape to the world map
    new = np.empty(world.map.shape[:2], dtype=np.str_)

    # Iterate through the world and assign the appearance of the object at that location
    for index, _ in np.ndenumerate(world.map[:, :, 0]):
        H, W = index
        # iterate from top to bottom
        for L in reversed(range(world.map.shape[2])):
            # if the entity is not empty, get its appearance, and we don't need to check the lower layers.
            if world.map[H, W, L].kind != "EmptyEntity":
                new[H, W] = entity_map[world.map[H, W, L].kind]
                break
            # continue to check the lower layers if the entity is not empty.
            else:
                new[H, W] = entity_map[world.map[H, W, L].kind]

    # If no location, return the full visual field
    if location is None or vision is None:
        return new.astype(np.str_)

    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.subtract(
            [world.map.shape[0] // 2, world.map.shape[1] // 2], location[0:2]
        )

        # Shift the array, and fill the appearances of coordinates outside the map with the fill entity's appearance.
        new = shift(array=new, shift=shift_dims, cval=entity_map[fill_entity_kind])

        # Set up the dimensions of the array to crop
        crop_h = (
            world.map.shape[0] // 2 - vision,
            world.map.shape[0] // 2 + vision + 1,
        )
        crop_w = (
            world.map.shape[1] // 2 - vision,
            world.map.shape[1] // 2 + vision + 1,
        )
        # Crop the array to the selected dimensions
        new = new[slice(*crop_h), slice(*crop_w)]

        # Return the agent's sliced observation space
        return new.astype(np.str_)
