import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2  # for generating random shapes
from PIL import Image, ImageDraw
import random


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Initial CNN Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # First Transformer
        self.transformer_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=1
        )

        # CNN Layer after first Transformer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Second Transformer
        self.transformer_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1
        )

        # Final layers
        self.fc_mu = nn.Linear(128 * 16 * 16, 256)  # Adjusted the input dimension
        self.fc_logvar = nn.Linear(128 * 16 * 16, 256)  # Adjusted the input dimension

    def forward(self, x):
        # First CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Reshape and add positional encoding for first Transformer
        x = x.flatten(2).permute(2, 0, 1)

        # First Transformer Encoder Layer
        x = self.transformer_encoder1(x)

        # Reshape back for CNN
        x = x.permute(1, 2, 0).reshape(x.size(1), 64, 32, 32)  # Adjusted the dimensions

        # Second CNN layer after Transformer
        x = F.relu(self.bn3(self.conv3(x)))

        # Reshape and add positional encoding for second Transformer
        x = x.flatten(2).permute(2, 0, 1)

        # Second Transformer Encoder Layer
        x = self.transformer_encoder2(x)

        # Flatten and Final layers
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)  # Flatten

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Middle(nn.Module):
    def __init__(self):
        super(Middle, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.1)  # 10% dropout
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.1)  # 10% dropout

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.dropout1(z)  # Apply dropout after activation
        z = F.relu(self.fc2(z))
        z = self.dropout2(z)  # Apply dropout after activation
        return z


import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Fully connected layer
        self.fc = nn.Linear(
            256, 128 * 16 * 16
        )  # Adjusted to match the output of the encoder

        # First deconvolution layer
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Second deconvolution layer
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        # Third deconvolution layer
        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(16)

        # Final convolution layer to get to the same size as the input image
        self.deconv4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(
            z.size(0), 128, 16, 16
        )  # Adjusted to match the output of the encoder

        z = F.relu(self.bn1(self.deconv1(z)))
        z = F.relu(self.bn2(self.deconv2(z)))
        z = F.relu(self.bn3(self.deconv3(z)))
        z = torch.sigmoid(self.deconv4(z))  # Output values are in [0, 1] range
        return z


class VAE(nn.Module):
    def __init__(self, num_classes=4):  # Assuming 4 classes for letters A, b, d, E
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.middle = Middle()
        self.classifier_head = nn.Linear(256, num_classes)  # Classification head

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = self.middle(z)
        recon = self.decoder(z)
        class_output = F.softmax(self.classifier_head(z), dim=1)
        return recon, mu, logvar, class_output


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
        shifted = False
        if shifted == True:
            shift_value = torch.randint(0, 7, (1,)).item()
        else:
            shift_value = 0
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


def draw_alien(color_choice):
    IMAGE_SIZE = (128, 128)
    if color_choice == "green":
        SKIN_COLORS = (50, 255, 50)
    if color_choice == "grey":
        SKIN_COLORS = (150, 150, 150)
    if color_choice == "yellow":
        SKIN_COLORS = (255, 255, 150)

    # SKIN_COLORS = [(50, 255, 50), (150, 150, 150), (255, 255, 150)]  # Green, Grey, Yellow

    # Create blank image
    img = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
    d = ImageDraw.Draw(img)

    # Randomly choose skin color
    # skin_color = random.choice(SKIN_COLORS)
    skin_color = SKIN_COLORS

    # Body
    body_shape = random.choice(["round", "elongated", "amorphous"])
    if body_shape == "round":
        d.ellipse([(32, 32), (96, 96)], fill=skin_color)
    elif body_shape == "elongated":
        d.ellipse([(32, 16), (96, 112)], fill=skin_color)
    elif body_shape == "amorphous":
        d.polygon([(32, 32), (64, 16), (96, 32), (64, 96)], fill=skin_color)

    # Eyes
    num_eyes = random.choice([1, 2, 3, 4])
    eye_colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255)]
    for i in range(num_eyes):
        x = 48 + (i % 2) * 32
        y = 48 + (i // 2) * 32
        d.ellipse([(x - 8, y - 8), (x + 8, y + 8)], fill=random.choice(eye_colors))

    # Antennae
    num_antennae = random.choice([0, 1, 2])
    for i in range(num_antennae):
        x = 48 + i * 32
        d.line([(x, 16), (x, 32)], fill=skin_color, width=2)

    # Tail
    if random.choice([True, False]):
        tail_color = random.choice(SKIN_COLORS)
        d.line([(64, 96), (64, 112)], fill=tail_color, width=4)

    return img


def generate_random_aliens_with_labels(num_images, letter_probs=None):
    images = np.zeros((num_images, 128, 128, 3), dtype=np.uint8)
    labels = []
    colors = []
    letter_to_label = {"A": 0, "b": 1, "d": 2, "E": 3}
    default_probs = [0.25, 0.25, 0.25, 0.25]

    for i in range(num_images):
        # Skin color
        color_choice = random.choice(["green", "grey", "yellow"])
        colors.append(color_choice)

        img = draw_alien(color_choice)
        img_array = np.array(img)
        images[i] = img_array

        # Letter
        if letter_probs and color_choice in letter_probs:
            probs = letter_probs[color_choice]
        else:
            probs = default_probs
        # letter_choice = random.choice(["A", "b", "d", "E"], p=probs)
        letter_choice = random.choice(["A", "b", "d", "E"])
        labels.append(letter_to_label[letter_choice])

        # Add the letter to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        images[i] = cv2.putText(
            images[i],
            letter_choice,
            (10, 20),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    images = images / 255.0
    return (
        torch.tensor(images.transpose((0, 3, 1, 2)), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        colors,
    )


def generate_random_aliens_with_labels_old(num_images, letter_probs=None):
    images = np.zeros((num_images, 16, 16, 3))
    labels = []  # List to store the labels
    colors = []
    letter_to_label = {
        "A": 0,
        "b": 1,
        "d": 2,
        "E": 3,
    }  # Mapping letters to integer labels
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

        # Add the label for this letter to the labels list
        labels.append(letter_to_label[letter_choice])
        colors.append(color_choice)

        # Add the letter to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        images[i] = cv2.putText(
            images[i], letter_choice, (0, 4), font, 0.2, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Badge
        badge_code = np.random.choice([0, 1], size=8)
        badge_str = "".join(map(str, badge_code))
        font = cv2.FONT_HERSHEY_SIMPLEX
        images[i] = cv2.putText(
            images[i], badge_str, (4, 14), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA
        )

    images = images / 255.0
    return (
        torch.tensor(images.transpose((0, 3, 1, 2)), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        colors,
    )


# Loss Function
def loss_function(recon_x, x, mu, logvar, class_output, class_target, learn_labels=0.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    classification_loss = F.cross_entropy(class_output, class_target)
    return BCE + KLD + learn_labels * classification_loss


# Training Function


def train_vae(vae, dataloader, num_epochs, eval_sync=100):
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        for batch in dataloader:
            (
                masked_data,
                unmasked_data,
                class_target,
            ) = batch  # Assuming each batch contains masked, unmasked data, and class labels
            optimizer.zero_grad()
            recon_batch, mu, logvar, class_output = vae(masked_data)
            loss = loss_function(
                recon_batch, unmasked_data, mu, logvar, class_output, class_target
            )
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % eval_sync == 0:
            images, class_labels, colors = generate_random_aliens_with_labels(
                15, letter_probs
            )  # Generating 15 new images for evaluation
            # Assuming you also generate ground truth labels for evaluation_images
            visualize_reconstruction(letter_predictor, vae, images, num_samples=15)
            eval_dat = evaluate_model(
                letter_predictor, vae, letter_probs, num_samples=10000
            )
            print(eval_dat)


import matplotlib.pyplot as plt


def visualize_reconstruction(letter_predictor, vae, images, num_samples=5):
    with torch.no_grad():  # Disable gradient computation
        masked_images = apply_mask(images)
        print(f"Shape of masked_images: {masked_images.shape}")
        reconstructed_images, _, _, _ = vae(masked_images)
        print(f"Shape of reconstructed_images: {reconstructed_images.shape}")
        top5_lines = reconstructed_images[:, :, :5, :]
        letter_guess = letter_predictor(top5_lines)

        # Assuming letter_guess is the output from a softmax layer
        probabilities = torch.softmax(letter_guess, dim=1).numpy()

    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    rescale_factor = 1.0
    label_map = {
        0: "A",
        1: "b",
        2: "d",
        3: "E",
    }  # Mapping from integer labels to letters

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

        # Plotting the softmax probabilities as a histogram
        axs[i, 3].bar(range(4), probabilities[i])
        axs[i, 3].set_xticks(range(4))
        axs[i, 3].set_xticklabels([label_map[j] for j in range(4)])
        axs[i, 3].set_title(f"Letter Guess Probabilities")

        # Display the guessed letter
        guessed_letter = label_map[np.argmax(probabilities[i])]
        axs[i, 3].text(1.5, 0.8, f"Guessed: {guessed_letter}", fontsize=12, ha="center")

    plt.tight_layout()
    plt.show()


import pandas as pd


def evaluate_model(letter_predictor, vae, letter_probs, num_samples=10000):
    # Generate aliens
    images, class_labels, colors = generate_random_aliens_with_labels(
        num_samples, letter_probs
    )

    # Apply mask sometimes
    masked_aliens = apply_mask(images)

    # Run through the VAE model
    with torch.no_grad():
        reconstructed_images, _, _, _ = vae(masked_aliens)
        top5_lines = reconstructed_images[:, :, :5, :]

    # Run through the LetterPredictor model
    with torch.no_grad():
        letter_guess = letter_predictor(top5_lines)

    # Get softmax probabilities and guessed letters
    probabilities = torch.softmax(letter_guess, dim=1).numpy()
    guessed_letters = np.argmax(probabilities, axis=1)

    # Create DataFrame to store results
    df = pd.DataFrame(
        {
            "Color": colors,
            "True_Letter": class_labels,
            "Guessed_Letter": guessed_letters,
        }
    )

    # Create summary table
    summary_tables = {}
    for color in set(colors):
        subset_df = df[df["Color"] == color]
        summary_table = pd.crosstab(
            subset_df["True_Letter"],
            subset_df["Guessed_Letter"],
            margins=True,
            margins_name="Total",
        )
        summary_tables[color] = summary_table

    return summary_tables


class LetterPredictor(nn.Module):
    def __init__(self):
        super(LetterPredictor, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # Adjust parameters as needed 10240
        # self.fc1 = nn.Linear(
        #    16 * 128 * 128, 128
        # )  # Adjust dimensions based on the output of conv1 and image size
        self.fc1 = nn.Linear(10240, 128)

        self.fc2 = nn.Linear(128, 4)  # Assuming 4 classes for letters A, b, d, E

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, vae, dataloader, num_epochs=10):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch in dataloader:
                masked_data, _, class_target = batch
                reconstructed_images, _, _, _ = vae(masked_data)
                top5_lines = reconstructed_images[:, :, :5, :]
                optimizer.zero_grad()
                outputs = self(top5_lines)
                loss = criterion(outputs, class_target)
                loss.backward()
                optimizer.step()
            print(f"Letter Predictor: Epoch {epoch}, Loss: {loss.item()}")


# ------- BASELINE LETTER PREDICTOR ---------------------

# Initialize and train VAE without confounds to get good letter representations
vae = VAE()
letter_predictor = LetterPredictor()

# Generate some synthetic data for testing
letter_probs = {
    "red": [0.25, 0.25, 0.25, 0.25],
    "green": [0.25, 0.25, 0.25, 0.25],
    "blue": [0.25, 0.25, 0.25, 0.25],
}

num_images = 1000
images, class_labels, colors = generate_random_aliens_with_labels(
    num_images, letter_probs
)  # Assuming this function also returns labels
masked_images = apply_mask(images)

# Prepare data for training
dataset = TensorDataset(
    images, images, class_labels
)  # should be trained on unmasked images for the letter_predictor
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for repeats in range(3):
    train_vae(vae, dataloader, num_epochs=500, eval_sync=10)  # Training
    letter_predictor.train_model(vae, dataloader, num_epochs=100)

# ------- CONFOUNDED ALIEN WORLD ---------------------

# new VAE with new aliens and old letter predictor

vae = VAE()

for reps in range(10):
    # Generate some synthetic data for testing
    letter_probs = {
        "red": [0.05, 0.7, 0.2, 0.05],
        "green": [0.05, 0.2, 0.7, 0.05],
        "blue": [0.25, 0.25, 0.25, 0.25],
    }

    num_images = 1000
    images, class_labels, colors = generate_random_aliens_with_labels(
        num_images, letter_probs
    )  # Assuming this function also returns labels
    masked_images = apply_mask(images)

    # Prepare data for training
    dataset = TensorDataset(
        masked_images, images, class_labels
    )  # Passing masked, unmasked images, and class labels
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_vae(vae, dataloader, num_epochs=500, eval_sync=100)  # Training
