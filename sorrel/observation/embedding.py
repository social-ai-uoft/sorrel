import matplotlib.pyplot as plt
import numpy as np

from sorrel.location import Location
from sorrel.worlds import Gridworld


def positional_embedding(
    location: tuple | Location, world: Gridworld, scale: tuple[int, int]
) -> np.ndarray:
    """Get the embedding value for a location within an environment.

    Args:
        location: (tuple | Location) The location to be embedded.
        world: (Gridworld) The gridworld environment within which embeddings should be computed.
        scale: (tuple[int, int]) The scale for encoding coordinates in the X and Y dimensions.

    Returns:
        np.ndarray: The positional embedding for the given location, with a shape of `(1, (scale[0] + scale[1]) * 2)`.

    .. note:: A lower value for :attr:`scale[0]` or :attr:`scale[1]` results in a lower resolution. This can mean that embedding values repeat, meaning that locations are not uniquely identified. Higher values of :attr:`scale[0]` or :attr:`scale[1]` result in a longer positional embedding value, but are able to uniquely encode points on a larger grid.
    """

    # Initialize embedding list for the given coordinate (x, y)
    x, y = location[0:2]
    grid_size = world.map.shape[0:2]
    embedding = []

    # Encoding for x dimension at different resolutions
    for i in range(scale[0]):
        freq_x = (
            2 * np.pi * (2**i) / grid_size[0]
        )  # Frequency increases with each scale
        embedding.append(np.sin(freq_x * x))
        embedding.append(np.cos(freq_x * x))

    # Encoding for y dimension at different resolutions
    for j in range(scale[0]):
        freq_y = (
            2 * np.pi * (2**j) / grid_size[1]
        )  # Frequency increases with each scale
        embedding.append(np.sin(freq_y * y))
        embedding.append(np.cos(freq_y * y))

    return np.array(embedding)


def generate_positional_embedding(
    grid_size: tuple[int, int], scale: tuple[int, int]
) -> np.ndarray:
    """Create an array of positional embeddings for all points on a grid.

    Args:
        grid_size: (tuple[int, int]) A tuple indicating the size of the X and Y axes of the grid.
        scale: (tuple[int, int]) The scale for encoding coordinates in the X and Y dimensions.

    Returns:
        np.ndarray: An array of the embedding values for all points on the grid.
    """
    # Initialize positional embeddings matrix
    embeddings = np.zeros((grid_size[0], grid_size[1], 2 * (scale[0] + scale[1])))

    # Generate positional encodings for each resolution
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            embedding = []

            # Encoding for x dimension at different resolutions
            for i in range(scale[0]):
                freq_x = (
                    2 * np.pi * (2**i) / grid_size[0]
                )  # Frequency increases with each scale
                embedding.append(np.sin(freq_x * x))
                embedding.append(np.cos(freq_x * x))

            # Encoding for y dimension at different resolutions
            for j in range(scale[1]):
                freq_y = (
                    2 * np.pi * (2**j) / grid_size[1]
                )  # Frequency increases with each scale
                embedding.append(np.sin(freq_y * y))
                embedding.append(np.cos(freq_y * y))

            embeddings[x, y] = embedding

    return embeddings


def recover_coordinates(
    embedding: np.ndarray, grid_size: tuple[int, int], scale: tuple[int, int]
) -> np.ndarray:
    """Recover coordinates by finding the closest matching embedding.

    Args:
        embedding: The positional embedding of all locations in a grid.
        grid_size: (tuple[int, int]) A tuple indicating the size of the X and Y axes of the grid.
        scale: (tuple[int, int]) The scale for encoding coordinates in the X and Y dimensions.

    Returns:
        np.ndarray: The recovered coordinates.

    .. warning:: This function expects all embeddings in the grid. If a single embedding or partial list is input, the function will fail.
    """

    embedding = embedding.reshape(np.prod(grid_size), -1)
    all_embeddings = generate_positional_embedding(grid_size, scale).reshape(
        np.prod(grid_size), -1
    )
    grid_positions = np.array(
        [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]
    ).reshape(np.prod(grid_size), -1)
    recovered_coordinates = np.zeros((grid_size[0], grid_size[1], 2))

    for each_embedding, (x, y) in zip(embedding, grid_positions):
        distances = np.linalg.norm(all_embeddings - each_embedding, axis=1)
        recovered_idx = np.argmin(distances)
        recovered_coordinates[x, y] = [
            recovered_idx // grid_size[0],
            recovered_idx % grid_size[1],
        ]

    return recovered_coordinates


def test_embeddings(
    grid_size: tuple[int, int] = (40, 40), scale: tuple[int, int] = (4, 4)
) -> None:
    """Helper function to graph embeddings for a given embedding size/scale.

    Args:
        grid_size: (tuple[int, int]) A tuple indicating the size of the X and Y axes of the grid.
        scale: (tuple[int, int]) The scale for encoding coordinates in the X and Y dimensions.
    """

    # Generate positional embeddings for each grid square
    positional_embeddings = generate_positional_embedding(grid_size, scale)

    # Recover x, y coordinates from embeddings
    recovered_coords = recover_coordinates(positional_embeddings, grid_size, scale)

    # Visualize recovered coordinates
    plt.figure(figsize=(6, 6))
    plt.title("Recovered Coordinates from Positional Embedding")
    plt.scatter(
        *zip(*[(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]),
        c="red",
        label="Original Positions",
        alpha=0.5,
    )
    plt.scatter(
        recovered_coords[:, :, 0].flatten(),
        recovered_coords[:, :, 1].flatten(),
        c="blue",
        marker="x",
        label="Recovered Positions",
        alpha=0.5,
    )
    plt.legend()
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show()

    # Plotting the sine and cosine waves at different scales to visualize the frequencies

    plt.figure(figsize=(12, 8))

    # Plot sine and cosine for x-dimension
    for i in range(scale[0]):
        freq_x = 2 * np.pi * (2**i) / grid_size[0]
        x_vals = np.arange(grid_size[0])
        sin_x = np.sin(freq_x * x_vals)
        cos_x = np.cos(freq_x * x_vals)

        plt.subplot(scale[0], 2, 2 * i + 1)
        plt.plot(x_vals, sin_x, label=f"Sin, Scale {i+1} (Freq {freq_x:.2f})")
        plt.title(f"Sine Wave for X Dimension at Scale {i+1}")
        plt.xlabel("X")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.subplot(scale[0], 2, 2 * i + 2)
        plt.plot(
            x_vals, cos_x, label=f"Cos, Scale {i+1} (Freq {freq_x:.2f})", color="orange"
        )
        plt.title(f"Cosine Wave for X Dimension at Scale {i+1}")
        plt.xlabel("X")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))

    # Plot sine and cosine for y-dimension
    for j in range(scale[1]):
        freq_y = 2 * np.pi * (2**j) / grid_size[1]
        y_vals = np.arange(grid_size[1])
        sin_y = np.sin(freq_y * y_vals)
        cos_y = np.cos(freq_y * y_vals)

        plt.subplot(scale[1], 2, 2 * j + 1)
        plt.plot(y_vals, sin_y, label=f"Sin, Scale {j+1} (Freq {freq_y:.2f})")
        plt.title(f"Sine Wave for Y Dimension at Scale {j+1}")
        plt.xlabel("Y")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.subplot(scale[1], 2, 2 * j + 2)
        plt.plot(
            y_vals, cos_y, label=f"Cos, Scale {j+1} (Freq {freq_y:.2f})", color="orange"
        )
        plt.title(f"Cosine Wave for Y Dimension at Scale {j+1}")
        plt.xlabel("Y")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()
