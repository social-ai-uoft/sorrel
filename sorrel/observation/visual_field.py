# Import base packages
import numpy as np

# sorrel imports
from sorrel.worlds import Gridworld


def _build_one_hot(
    map_region: np.ndarray, entity_map: dict[str, np.ndarray], num_channels: int
) -> np.ndarray:
    """Build and layer-sum a one-hot appearance array over `map_region`."""
    arr = np.stack(
        [np.zeros_like(map_region, dtype=np.float64) for _ in range(num_channels)],
        axis=0,
    )
    for index, x in np.ndenumerate(map_region):
        arr[:, *index] = entity_map[x.kind]
    return np.sum(arr, axis=-1)


def _build_ascii(map_region: np.ndarray, entity_map: dict[str, str]) -> np.ndarray:
    """Build an ascii appearance array over `map_region`, taking the topmost non-empty
    entity per cell."""
    new = np.empty(map_region.shape[:2], dtype=np.str_)
    for index, _ in np.ndenumerate(map_region[:, :, 0]):
        H, W = index
        for L in reversed(range(map_region.shape[2])):
            kind = map_region[H, W, L].kind
            new[H, W] = entity_map[kind]
            if kind != "EmptyEntity":
                break
    return new


def _window_bounds(
    location: tuple, vision: int, map_shape: tuple[int, int, int]
) -> tuple[int, int, int, int, int, int, int]:
    """Compute clamped map bounds and destination offset for a vision window.

    Returns:
        A tuple `(h_lo, h_hi, w_lo, w_hi, dest_h, dest_w, window)`.
    """
    loc_h, loc_w = location[0], location[1]
    map_h, map_w, _ = map_shape
    h_lo, h_hi = max(0, loc_h - vision), min(map_h, loc_h + vision + 1)
    w_lo, w_hi = max(0, loc_w - vision), min(map_w, loc_w + vision + 1)
    window = 2 * vision + 1
    dest_h, dest_w = h_lo - (loc_h - vision), w_lo - (loc_w - vision)
    return h_lo, h_hi, w_lo, w_hi, dest_h, dest_w, window


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
        `(number of channels, world.height, world.width)`.
        Here, the number channels is determined based on the one-hot entity map provided.
    """
    # Get the number of channels used by the model.
    num_channels = len(list(entity_map.values())[0])

    # If no location, return the full visual field
    if location is None or vision is None:
        new = _build_one_hot(world.map, entity_map, num_channels)
        return new.astype(np.float64)

    # Otherwise...
    else:
        # Only process the region of the map that the window actually covers,
        # instead of building and shifting a full-size array.
        h_lo, h_hi, w_lo, w_hi, dest_h, dest_w, window = _window_bounds(
            location, vision, world.map.shape
        )

        sub_map = world.map[h_lo:h_hi, w_lo:w_hi]
        sub = _build_one_hot(sub_map, entity_map, num_channels)

        new = np.empty((num_channels, window, window), dtype=np.float64)
        new[...] = np.asarray(entity_map[fill_entity_kind], dtype=np.float64)[
            :, None, None
        ]
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
    # If no location, return the full visual field
    if location is None or vision is None:
        new = _build_ascii(world.map, entity_map)
        return new.astype(np.str_)

    # Otherwise...
    else:
        # Only process the region of the map that the window actually covers,
        # instead of building and shifting a full-size array.
        h_lo, h_hi, w_lo, w_hi, dest_h, dest_w, window = _window_bounds(
            location, vision, world.map.shape
        )

        sub_map = world.map[h_lo:h_hi, w_lo:w_hi]
        sub = _build_ascii(sub_map, entity_map)

        new = np.full((window, window), entity_map[fill_entity_kind], dtype=np.str_)
        new[dest_h : dest_h + (h_hi - h_lo), dest_w : dest_w + (w_hi - w_lo)] = sub

        # Return the agent's sliced observation space
        return new.astype(np.str_)
