import numpy as np
import matplotlib.pyplot as plt

from sorrel.environments import GridworldEnv
from sorrel.location import Location

def positional_embedding(
        location: tuple | Location, 
        env: GridworldEnv, 
        num_res_x: int, 
        num_res_y: int):
    """
    Get the embedding value for a location within an environment.

    Args:
        location: A 

    """

    # Initialize embedding list for the given coordinate (x, y)
    x = location[0]
    y = location[1]
    x_dim = env.world.shape[0]
    y_dim = env.world.shape[1]
    embedding = []

    # Encoding for x dimension at different resolutions
    for i in range(num_res_x):
        freq_x = 2 * np.pi * (2**i) / x_dim  # Frequency increases with each scale
        embedding.append(np.sin(freq_x * x))
        embedding.append(np.cos(freq_x * x))

    # Encoding for y dimension at different resolutions
    for j in range(num_res_y):
        freq_y = 2 * np.pi * (2**j) / y_dim  # Frequency increases with each scale
        embedding.append(np.sin(freq_y * y))
        embedding.append(np.cos(freq_y * y))

    return np.array(embedding)


def generate_positional_embedding(x_dim, y_dim, num_res_x, num_res_y):
    # Initialize positional embeddings matrix
    embeddings = np.zeros((x_dim, y_dim, 2 * (num_res_x + num_res_y)))

    # Generate positional encodings for each resolution
    for x in range(x_dim):
        for y in range(y_dim):
            embedding = []

            # Encoding for x dimension at different resolutions
            for i in range(num_res_x):
                freq_x = (
                    2 * np.pi * (2**i) / x_dim
                )  # Frequency increases with each scale
                embedding.append(np.sin(freq_x * x))
                embedding.append(np.cos(freq_x * x))

            # Encoding for y dimension at different resolutions
            for j in range(num_res_y):
                freq_y = (
                    2 * np.pi * (2**j) / y_dim
                )  # Frequency increases with each scale
                embedding.append(np.sin(freq_y * y))
                embedding.append(np.cos(freq_y * y))

            embeddings[x, y] = embedding

    return embeddings


def recover_coordinates(embedding, x_dim, y_dim, num_res_x, num_res_y):
    # Recover coordinates by finding the closest matching embedding
    recovered_coordinates = np.zeros((x_dim, y_dim, 2))
    grid_positions = [(x, y) for x in range(x_dim) for y in range(y_dim)]
    flat_embeddings = embedding.reshape(x_dim * y_dim, -1)

    for idx, (x, y) in enumerate(grid_positions):
        recovered_coordinates[x, y] = [x, y]
        distances = np.linalg.norm(flat_embeddings - embedding[x, y], axis=1)
        recovered_idx = np.argmin(distances)
        recovered_coordinates[x, y] = [recovered_idx // y_dim, recovered_idx % y_dim]

    return recovered_coordinates


def test_embeddings(x_dim=40, y_dim=40, num_res_x=4, num_res_y=4):
    # Define the grid size and number of resolutions
    # x_dim, y_dim = 40, 40
    # num_res_x, num_res_y = 4, 4  # Number of resolutions/scales for x and y dimensions

    # Generate positional embeddings for each grid square
    positional_embeddings = generate_positional_embedding(
        x_dim, y_dim, num_res_x, num_res_y
    )

    # Recover x, y coordinates from embeddings
    recovered_coords = recover_coordinates(
        positional_embeddings, x_dim, y_dim, num_res_x, num_res_y
    )

    # Visualize recovered coordinates
    plt.figure(figsize=(6, 6))
    plt.title("Recovered Coordinates from Positional Embedding")
    plt.scatter(
        *zip(*[(x, y) for x in range(x_dim) for y in range(y_dim)]),
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
    for i in range(num_res_x):
        freq_x = 2 * np.pi * (2**i) / x_dim
        x_vals = np.arange(x_dim)
        sin_x = np.sin(freq_x * x_vals)
        cos_x = np.cos(freq_x * x_vals)

        plt.subplot(num_res_x, 2, 2 * i + 1)
        plt.plot(x_vals, sin_x, label=f"Sin, Scale {i+1} (Freq {freq_x:.2f})")
        plt.title(f"Sine Wave for X Dimension at Scale {i+1}")
        plt.xlabel("X")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.subplot(num_res_x, 2, 2 * i + 2)
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
    for j in range(num_res_y):
        freq_y = 2 * np.pi * (2**j) / y_dim
        y_vals = np.arange(y_dim)
        sin_y = np.sin(freq_y * y_vals)
        cos_y = np.cos(freq_y * y_vals)

        plt.subplot(num_res_y, 2, 2 * j + 1)
        plt.plot(y_vals, sin_y, label=f"Sin, Scale {j+1} (Freq {freq_y:.2f})")
        plt.title(f"Sine Wave for Y Dimension at Scale {j+1}")
        plt.xlabel("Y")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.subplot(num_res_y, 2, 2 * j + 2)
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
