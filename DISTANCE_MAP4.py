from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray, gray2rgb
import numpy as np
import cv2

# Load real and synthetic images
image_pairs = [("data/membrane/IMAGES/r/real_1.png", "data/membrane/IMAGES/f/fake_1.png"),
               ("data/membrane/IMAGES/r/real_2.png", "data/membrane/IMAGES/f/fake_2.png"),
               ("data/membrane/IMAGES/r/real_3.png", "data/membrane/IMAGES/f/fake_3.png"),
               ("data/membrane/IMAGES/r/real_4.png", "data/membrane/IMAGES/f/fake_4.png"),
               ("data/membrane/IMAGES/r/real_5.png", "data/membrane/IMAGES/f/fake_5.png"),
               ("data/membrane/IMAGES/r/real_6.png", "data/membrane/IMAGES/f/fake_6.png"),
               ("data/membrane/IMAGES/r/real_7.png", "data/membrane/IMAGES/f/fake_7.png"),
               ("data/membrane/IMAGES/r/real_8.png", "data/membrane/IMAGES/f/fake_8.png"),
               ("data/membrane/IMAGES/r/real_9.png", "data/membrane/IMAGES/f/fake_9.png"),
               ("data/membrane/IMAGES/r/real_10.png", "data/membrane/IMAGES/f/fake_10.png")]

# Create a blank canvas to merge the visual distance maps
canvas_height = 0
canvas_width = 0
for real_file, synthetic_file in image_pairs:
    real_image = cv2.imread(real_file)
    synthetic_image = cv2.imread(synthetic_file)
    canvas_height = max(canvas_height, real_image.shape[0])
    canvas_width += real_image.shape[1]
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Generate the visual distance map for each image pair and merge into the canvas
x = 0
for real_file, synthetic_file in image_pairs:
    # Load real and synthetic images
    real_image = cv2.imread(real_file)
    synthetic_image = cv2.imread(synthetic_file)

    # Resize
    real_image = cv2.resize(real_image, (512,512))
    synthetic_image = cv2.resize(synthetic_image, (real_image.shape[1], real_image.shape[0]))

    # Grayscale
    g1 = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2GRAY)

    # Find the difference between two images
    # Compute the mean structural similarity index
    (similar, ssim_map) = ssim(g1,g2,full=True)

    # Rescale SSIM distance map to 0-1 range
    ssim_map = (ssim_map - np.min(ssim_map)) / (np.max(ssim_map) - np.min(ssim_map))

    # Convert SSIM distance map to RGB color map
    ssim_map_rgb = gray2rgb(ssim_map)

    # Threshold SSIM distance map to highlight dissimilar areas in red
    ssim_threshold = np.mean(ssim_map)
    ssim_map_red = np.zeros((ssim_map.shape[0], ssim_map.shape[1], 3), dtype=np.uint8)
    ssim_map_red[:, :, 0] = np.where(ssim_map < ssim_threshold, np.round((ssim_threshold - ssim_map) * 255 / ssim_threshold), 0)
    ssim_map_red[:, :, 1] = 0
    ssim_map_red[:, :, 2] = 255

    # Blend color maps with real image
    blended_image = 0.5 * img_as_ubyte(real_image) + 0.5 * ssim_map_red + 0.5 * ssim_map_rgb.astype(np.uint8)

    # Merge the blended image into the canvas
    canvas[0:blended_image.shape[0], x:x+blended_image.shape[1], :] = blended_image

    # Update x position for next image
    x += blended_image.shape[1]

# Save the merged visual distance map as a PNG file
io.imsave("distance_map/visual_distance_map.png", canvas)