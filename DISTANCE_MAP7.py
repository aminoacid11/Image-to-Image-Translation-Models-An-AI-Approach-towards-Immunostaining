import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Create a list of image paths
real_img_paths = ['data/membrane/IMAGES/r/real_{}.png'.format(i) for i in range(1, 11)]
synth_img_paths = ['data/membrane/IMAGES/f/fake_{}.png'.format(i) for i in range(1, 11)]

# Create an empty list to store the output images
output_images = []

# Loop over each image pair and compute the visual distance map
for i in range(10):
    # Read the real and synthetic images
    real_img = cv2.imread(real_img_paths[i])
    synth_img = cv2.imread(synth_img_paths[i])

    # Convert images to grayscale
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    synth_gray = cv2.cvtColor(synth_img, cv2.COLOR_BGR2GRAY)

    # Compute structural similarity index
    ssim_index, ssim_img = ssim(real_gray, synth_gray, full=True)

    # Scale the similarity index to the range [0, 255] and convert to uint8
    heatmap = cv2.normalize(ssim_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply the colormap (jet) to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend the heatmap with the original real image using alpha blending
    alpha = 0.5
    output = cv2.addWeighted(real_img, alpha, heatmap, 1 - alpha, 0)

    # Resize the output image to 512x512
    output = cv2.resize(output, (512, 512))

    # Append the output image to the list
    output_images.append(output)

# Combine all output images horizontally into a single image
final_output = np.concatenate(output_images, axis=1)

# Display the final output image
cv2.imshow('Visual Distance Maps', final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()