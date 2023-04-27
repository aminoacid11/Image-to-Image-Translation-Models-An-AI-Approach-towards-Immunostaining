from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

img_num = 1
# Load images
real_image = cv2.imread("data/membrane/IMAGES/r/real_{}.png".format(img_num))
synthetic_image = cv2.imread("data/membrane/IMAGES/f/fake_{}.png".format(img_num))

# Resize
real_image = cv2.resize(real_image, (512,512))
synthetic_image = cv2.resize(synthetic_image, (real_image.shape[1], real_image.shape[0]))

# Grayscale
g1 = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2GRAY)

# Find the difference between two images
# Compute the mean structural similarity index
(similar, ssim_map) = ssim(g1,g2,full=True)

# Calculate SSIM distance map
# ssim_map = ssim(real_image, synthetic_image,multichannel=True)
print(ssim_map)

# Rescale SSIM distance map to 0-255 range
ssim_map = (ssim_map - np.min(ssim_map)) / (np.max(ssim_map) - np.min(ssim_map)) * 255
ssim_map = ssim_map.astype(np.uint8)

# Convert SSIM distance map to RGB color map
ssim_map_rgb = np.zeros((ssim_map.shape[0], ssim_map.shape[1], 3), dtype=np.uint8)
ssim_map_rgb[:, :, 0] = ssim_map
ssim_map_rgb[:, :, 1] = ssim_map
ssim_map_rgb[:, :, 2] = ssim_map

# Blend SSIM distance map with real image
blended_image = 0.5 * img_as_ubyte(real_image) + 0.5 * ssim_map_rgb

# Save blended image
io.imsave("distance_map1.png", blended_image)