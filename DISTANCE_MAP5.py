import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.metrics import structural_similarity as ssim
import cv2

# Define the paths to the real and synthetic images
real_path = "./data/membrane/IMAGES/r/"
synthetic_path = "./data/membrane/IMAGES/f/"

# Load the images and calculate the SSIM
image_pairs = []
ssim_values = []
original_images = []
number_of_images = 5
for i in range(1, number_of_images+1):
    original_image = cv2.imread("data/membrane/IMAGES/r/real_{}.png".format(i))
    real_image = img_as_float(color.rgb2gray(io.imread(os.path.join(real_path, f"real_{i}.png"))))
    synthetic_image = img_as_float(color.rgb2gray(io.imread(os.path.join(synthetic_path, f"fake_{i}.png"))))
    original_images.append(original_image)
    image_pairs.append((real_image, synthetic_image))
    ssim_values.append(ssim(real_image, synthetic_image))

# Create a figure with subplots for each image
fig, axes = plt.subplots(nrows=number_of_images, ncols=2, figsize=(10, 20))

# Loop over each image pair and plot the original and the visual distance map
for i, (real_image, synthetic_image) in enumerate(image_pairs):
    org_img = original_images[i]
    # Calculate the SSIM
    ssim_value = ssim_values[i]

    # Generate the visual distance map
    distance_map = np.abs(real_image - synthetic_image)
    distance_map = np.interp(distance_map, (0, np.max(distance_map)), (0, 1))
    distance_map = color.gray2rgb(distance_map)

    # Color the closer part of the images as blue and dissimilar part as red
    distance_map[distance_map[..., 0] >= 0.5] = [1, 0, 0]
    distance_map[distance_map[..., 0] < 0.5] = [0, 0, 1]

    # Plot the original image and the visual distance map in the subplot
    axes[i, 0].imshow(org_img)
    axes[i, 0].set_title(f"Real Image {i+1}",fontsize=8)
    axes[i, 1].imshow(distance_map)
    axes[i, 1].set_title(f"Visual Distance Map (SSIM: {ssim_value:.2f})",fontsize=8)
    axes[i, 1].axis("off")
    cbar = axes[i, 1].imshow(distance_map[..., 0], cmap="coolwarm")
    fig.colorbar(cbar, ax=axes[i, 1])

# Remove the x and y ticks from all subplots
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

# Add a common title for the figure
fig.suptitle("Visual Distance Maps using SSIM")

# Display the figure
plt.show()