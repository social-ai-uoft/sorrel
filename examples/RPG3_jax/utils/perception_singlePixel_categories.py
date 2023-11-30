import numpy as np
import torch


def agent_visualfield(
    world, location, k=4, wall_app=[50.0, 50.0, 50.0], num_channels=3
):
    """
    Create an agent visual field of size (2k + 1, 2k + 1) pixels
    Layer = location[2] and layer in the else are added to this function
    """

    if len(location) > 2:
        layer = location[2]
    else:
        layer = 0

    bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)
    # instantiate image
    images = []

    for channel in range(num_channels):
        image = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
        images.append(image)

    for i in range(bounds[0], bounds[1] + 1):
        for j in range(bounds[2], bounds[3] + 1):
            # while outside the world array index...
            if i < 0 or j < 0 or i >= world.shape[0] - 1 or j >= world.shape[1]:
                # image has shape bounds[1] - bounds[0], bounds[3] - bounds[2]
                # visual appearance = wall
                for channel in range(num_channels):
                    images[channel][i - bounds[0], j - bounds[2]] = wall_app[channel]

            else:
                for channel in range(num_channels):
                    images[channel][i - bounds[0], j - bounds[2]] = world[
                        i, j, layer
                    ].appearance[channel]

    # Composite image by interlacing the red, green, and blue channels, or one hots
    state = np.dstack(tuple(images))
    return state


def full_visualfield(env, depth, channels):
    (
        H,
        W,
        _,
    ) = (
        env.world.shape
    )  # Assuming env.world is a numpy ndarray or has a similar interface.
    tensor = torch.zeros((1, 1, channels, H, W))  # Initialize the tensor with zeros.

    for h in range(H):
        for w in range(W):
            appearance = env.world[h, w, depth].appearance
            tensor[0, 0, :, h, w] = torch.tensor(appearance)

    return tensor


def agent_visualfield_experimental(world, location, k=4, wall_app=[50.0, 50.0, 50.0]):
    """
    Create an agent visual field of size (2k + 1, 2k + 1) pixels
    Layer = location[2] and layer in the else are added to this function
    """
    if len(location) > 2:
        layer = location[2]
    else:
        layer = 0

    num_channels = world.shape[2]
    bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)

    # Instantiate visual field
    visual_field = np.zeros(
        (bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1, num_channels)
    )

    # Clip the bounds to the world boundaries
    clipped_bounds = (
        max(bounds[0], 0),
        min(bounds[1], world.shape[0] - 1),
        max(bounds[2], 0),
        min(bounds[3], world.shape[1] - 1),
    )

    # Assign appearances within the clipped bounds to the visual field
    visual_field[
        clipped_bounds[0] - bounds[0] : clipped_bounds[1] - bounds[0] + 1,
        clipped_bounds[2] - bounds[2] : clipped_bounds[3] - bounds[2] + 1,
    ] = world[
        clipped_bounds[0] : clipped_bounds[1] + 1,
        clipped_bounds[2] : clipped_bounds[3] + 1,
        layer,
    ]

    # Assign wall appearances to areas outside the clipped bounds
    visual_field[: clipped_bounds[0] - bounds[0], :] = wall_app
    visual_field[clipped_bounds[1] - bounds[0] + 1 :, :] = wall_app
    visual_field[:, : clipped_bounds[2] - bounds[2]] = wall_app
    visual_field[:, clipped_bounds[3] - bounds[2] + 1 :] = wall_app

    return visual_field
