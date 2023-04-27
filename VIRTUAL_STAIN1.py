import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

real_paths = []
stained_paths = []
for ind in range(5):
    real_paths.append('data/membrane/IMAGES/r/real_{}.png'.format(ind+1))
    stained_paths.append('data/membrane/IMAGES/f/fake_{}.png'.format(ind+1))

# Create a figure with 10 subplots
fig, axs = plt.subplots(5, 2, figsize=(8, 16))

# Loop through each pair of images and calculate the SSIM score
for i in range(len(real_paths)):
    # Load the real and virtually stained images
    real_img = cv2.imread(real_paths[i])
    stained_img = cv2.imread(stained_paths[i])

    # Convert the images to grayscale
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    stained_gray = cv2.cvtColor(stained_img, cv2.COLOR_BGR2GRAY)

    # Calculate the SSIM score between the real and stained images
    score = ssim(real_gray, stained_gray)

    # Plot the real and stained images side by side, with the SSIM score as the title
    axs[i, 0].imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
    axs[i, 1].imshow(cv2.cvtColor(stained_img, cv2.COLOR_BGR2RGB))
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')
    axs[i, 0].set_title('Real', fontsize=8, pad=5)
    axs[i, 1].set_title(f'Score: {score:.2f}\nStained', fontsize=8, pad=5)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
plt.show()