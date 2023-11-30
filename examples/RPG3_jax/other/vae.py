import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2  # for generating random shapes


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1
        )  # added stride=2
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # added stride=2
        self.bn2 = nn.BatchNorm2d(64)

        self.fc_mu = nn.Linear(
            64 * 4 * 4, 256
        )  # adjusted dimensions due to stride=2 convolutions
        self.fc_logvar = nn.Linear(
            64 * 4 * 4, 256
        )  # adjusted dimensions due to stride=2 convolutions

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Middle(nn.Module):
    def __init__(self):
        super(Middle, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.5)  # 25% dropout
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.35)  # 10% dropout

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.dropout1(z)  # Apply dropout after activation
        z = F.relu(self.fc2(z))
        z = self.dropout2(z)  # Apply dropout after activation
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(
            256, 64 * 4 * 4
        )  # No change here, but ensure the dimensions match the output of the encoder

        # Replace Upsample + Conv2d with ConvTranspose2d to allow learning during upsampling
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Adjust stride and output_padding
        self.bn1 = nn.BatchNorm2d(32)

        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Adjust stride and output_padding
        self.bn2 = nn.BatchNorm2d(16)

        self.deconv3 = nn.Conv2d(
            16, 3, kernel_size=3, stride=1, padding=1
        )  # This stays the same as it's not an upsampling layer

    def forward(self, z):
        z = self.fc(z)
        z = z.view(
            z.size(0), 64, 4, 4
        )  # Ensure the dimensions here match the output of the encoder
        z = F.relu(self.bn1(self.deconv1(z)))  # Apply BatchNorm and ReLU
        z = F.relu(self.bn2(self.deconv2(z)))  # Apply BatchNorm and ReLU
        z = torch.sigmoid(self.deconv3(z))  # Output values are in [0, 1] range
        return z

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 64, 4, 4)

        z = F.relu(self.bn1(self.deconv1(z)))
        z = F.relu(self.bn2(self.deconv2(z)))
        z = torch.sigmoid(self.deconv3(z))  # output values are in [0, 1] range
        return z


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.middle = Middle()  # Include middle layer

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = self.middle(z)  # Pass through middle layer
        return self.decoder(z), mu, logvar


def apply_mask(images, mask_size=3):
    masked_images = images.clone()
    for i in range(images.size(0)):
        h = torch.randint(0, images.size(2) - mask_size + 1, (1,)).item()
        w = torch.randint(0, images.size(3) - mask_size + 1, (1,)).item()

        random_gray_value = torch.rand(mask_size, mask_size) * 0.25
        for c in range(images.size(1)):  # Loop over color channels
            masked_images[
                i, c, h : h + mask_size, w : w + mask_size
            ] = random_gray_value

        # 50% chance to partially mask the top 5 rows
        if torch.rand(1).item() < 0.5:
            # Identify the white pixels in the top 5 rows
            white_pixels = (masked_images[i, :, 0:5, :] == 1.0).all(dim=0)
            num_white_pixels = white_pixels.sum().item()

            # Determine a random percentage of white pixels to mask
            mask_percentage = (
                0.5 + torch.rand(1).item() * 0.3
            )  # Random percentage between 50% and 80%
            num_pixels_to_mask = int(num_white_pixels * mask_percentage)

            # Randomly select white pixels to mask
            white_pixel_indices = white_pixels.nonzero(as_tuple=True)
            mask_indices = torch.randperm(num_white_pixels)[:num_pixels_to_mask]

            # Mask the selected white pixels with random grayscale values
            random_gray_value = torch.rand(num_pixels_to_mask) * 0.25
            for c in range(images.size(1)):  # Loop over color channels
                masked_images[i, c, 0:5, :][
                    white_pixel_indices[0][mask_indices],
                    white_pixel_indices[1][mask_indices],
                ] = random_gray_value

        # Shift white pixels to the right
        shift_value = torch.randint(0, 7, (1,)).item()
        shifted_image = torch.roll(
            masked_images[i, :, 0:5, :], shifts=shift_value, dims=2
        )
        masked_images[i, :, 0:5, :] = shifted_image

        # Apply random grayscale to all black pixels in the top 5 rows
        black_pixels = (masked_images[i, :, 0:5, :] == 0.0).all(dim=0)
        black_pixel_indices = black_pixels.nonzero(as_tuple=True)
        random_gray_value = torch.rand(black_pixels.sum().item()) * 0.25
        for c in range(images.size(1)):  # Loop over color channels
            masked_images[i, c, 0:5, :][black_pixel_indices] = random_gray_value

    return masked_images


def generate_random_shapes(num_images):
    images = np.zeros((num_images, 16, 16, 3))
    for i in range(num_images):
        color = (np.random.rand(), np.random.rand(), np.random.rand())
        if np.random.rand() > 0.5:
            center = (np.random.randint(5, 12), np.random.randint(5, 12))
            radius = np.random.randint(2, 5)
            images[i] = cv2.circle(np.zeros((16, 16, 3)), center, radius, color, -1)
        else:
            top_left = (np.random.randint(5, 12), np.random.randint(5, 12))
            bottom_right = (
                np.random.randint(top_left[0] + 1, 14),
                np.random.randint(top_left[1] + 1, 14),
            )
            images[i] = cv2.rectangle(
                np.zeros((16, 16, 3)), top_left, bottom_right, color, -1
            )
    return torch.tensor(
        images.transpose((0, 3, 1, 2)), dtype=torch.float32
    )  # change shape to [num_images, 3, 16, 16]


def generate_random_aliens(num_images, letter_probs=None):
    images = np.zeros((num_images, 16, 16, 3))
    default_probs = [0.25, 0.25, 0.25, 0.25]
    for i in range(num_images):
        # Skin color
        color_choice = np.random.choice(["red", "green", "blue"])
        if color_choice == "red":
            color = (255, 0, 0)
        elif color_choice == "green":
            color = (0, 255, 0)
        else:  # blue
            color = (0, 0, 255)

        # Body
        center = (8, 10)
        radius = 4
        images[i] = cv2.circle(np.zeros((16, 16, 3)), center, radius, color, -1)

        # Antennae
        antenna_length_choice = np.random.choice(["short", "medium", "long"])
        if antenna_length_choice == "short":
            antenna_length = 2
        elif antenna_length_choice == "medium":
            antenna_length = 4
        else:  # long
            antenna_length = 6
        images[i] = cv2.line(images[i], (8, 6), (8, 6 - antenna_length), (0, 0, 0), 1)

        # Eyes
        eye_choice = np.random.choice([1, 2])
        if eye_choice == 1:
            images[i] = cv2.circle(images[i], (8, 10), 1, (255, 255, 255), -1)
        else:  # 2 eyes
            images[i] = cv2.circle(images[i], (7, 10), 1, (255, 255, 255), -1)
            images[i] = cv2.circle(images[i], (9, 10), 1, (255, 255, 255), -1)

        # Arms
        arm_length_choice = np.random.choice(["short", "long"])
        arm_length = 2 if arm_length_choice == "short" else 4
        images[i] = cv2.line(images[i], (6, 12), (6, 12 + arm_length), (0, 0, 0), 1)
        images[i] = cv2.line(images[i], (10, 12), (10, 12 + arm_length), (0, 0, 0), 1)

        # Letter
        if letter_probs and color_choice in letter_probs:
            probs = letter_probs[color_choice]
        else:
            probs = default_probs
        letter_choice = np.random.choice(["A", "b", "d", "E"], p=probs)
        font = cv2.FONT_HERSHEY_SIMPLEX
        images[i] = cv2.putText(
            images[i], letter_choice, (0, 4), font, 0.2, (255, 255, 255), 1, cv2.LINE_AA
        )
        # print(color_choice, letter_choice)

        # Badge
        badge_code = np.random.choice([0, 1], size=8)
        badge_str = "".join(map(str, badge_code))
        font = cv2.FONT_HERSHEY_SIMPLEX
        images[i] = cv2.putText(
            images[i], badge_str, (4, 14), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA
        )
    images = images / 255.0
    return torch.tensor(images.transpose((0, 3, 1, 2)), dtype=torch.float32)


# Example usage:
# num_images = 10
# aliens = generate_random_aliens(num_images)


# Loss Function
def loss_function(recon_x, x, mu, logvar, beta=1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


# Training Function
def train_vae(vae, dataloader, num_epochs, eval_sync=100):
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        for batch in dataloader:
            (
                masked_data,
                unmasked_data,
            ) = batch  # Assuming each batch contains masked and unmasked data
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(masked_data)
            loss = loss_function(
                recon_batch, unmasked_data, mu, logvar
            )  # Using unmasked_data as the target
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        if epoch % eval_sync == 0:
            evaluation_images = generate_random_aliens(
                15, letter_probs
            )  # Generating 5 new images for evaluation
            visualize_reconstruction(vae, evaluation_images, num_samples=15)


import matplotlib.pyplot as plt


def visualize_reconstruction(vae, images, num_samples=5):
    with torch.no_grad():  # Disable gradient computation
        masked_images = apply_mask(images)
        reconstructed_images, _, _ = vae(masked_images)

    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    rescale_factor = 1.0

    for i in range(num_samples):
        axs[i, 0].imshow(images[i].permute(1, 2, 0) * rescale_factor)
        axs[i, 0].axis("off")
        axs[i, 0].set_title("Original Image")

        axs[i, 1].imshow(masked_images[i].permute(1, 2, 0) * rescale_factor)
        axs[i, 1].axis("off")
        axs[i, 1].set_title("Masked Image")

        axs[i, 2].imshow(reconstructed_images[i].permute(1, 2, 0) * rescale_factor)
        axs[i, 2].axis("off")
        axs[i, 2].set_title("Reconstructed Image")

    plt.show()


# Generate some synthetic data for testing
letter_probs = {
    "red": [0.0, 0.8, 0.2, 0.0],
    "green": [0.0, 0.2, 0.8, 0.0],
    "blue": [0.25, 0.25, 0.25, 0.25],
}

num_images = 1000
images = generate_random_aliens(num_images, letter_probs)
# images = generate_random_shapes(num_images)
masked_images = apply_mask(images)

# Prepare data for training

dataset = TensorDataset(
    masked_images, images
)  # Passing both masked and unmasked images
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Initialize and train VAE
vae = VAE()
epochs = 10000

train_vae(
    vae, dataloader, num_epochs=epochs, eval_sync=100
)  # Training for 10 epochs as an example

# Visualize the reconstruction on a new set of images
evaluation_images = generate_random_aliens(
    15, letter_probs
)  # Generating 5 new images for evaluation
visualize_reconstruction(vae, evaluation_images)


evaluation_images = generate_random_aliens(
    100, letter_probs
)  # Generating 5 new images for evaluation
visualize_reconstruction(vae, evaluation_images)
