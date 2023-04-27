from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

# Load real and synthetic images
image_pairs = [("data/membrane/IMAGES/r/real_{}.png".format(i),
                "data/membrane/IMAGES/f/fake_{}.png".format(i)) for i in range(1, 11)]

ind = 1
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
    print(ssim_map)

    # Rescale SSIM distance map to 0-1 range
    ssim_map = (ssim_map - np.min(ssim_map)) / (np.max(ssim_map) - np.min(ssim_map))

        # Convert SSIM distance map to RGB color map
    ssim_map_rgb = np.zeros((ssim_map.shape[0], ssim_map.shape[1], 3), dtype=np.uint8)
    ssim_map_rgb[:, :, 2] = np.round(ssim_map * 255) # blue channel

    # Threshold SSIM distance map to highlight dissimilar areas in red
    ssim_threshold = np.mean(ssim_map)
    ssim_map_red = np.zeros((ssim_map.shape[0], ssim_map.shape[1], 3), dtype=np.uint8)
    ssim_map_red[:, :, 0] = np.where(ssim_map < ssim_threshold, np.round((ssim_threshold - ssim_map) * 255 / ssim_threshold), 0)
    ssim_map_red[:, :, 1] = 0
    ssim_map_red[:, :, 2] = 255

    # Blend color maps with real image
    blended_image = 0.5 * img_as_ubyte(real_image) + 0.5 * ssim_map_red + 0.5 * ssim_map_rgb.astype(np.uint8)

    # Save blended image
    io.imsave("distance_map/distance_map_final{}.png".format(ind),blended_image)
    ind += 1