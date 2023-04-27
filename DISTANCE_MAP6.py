import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Read the real and synthetic images
real_img  = cv2.imread('data/membrane/IMAGES/r/real_1.png')
synth_img  = cv2.imread('data/membrane/IMAGES/f/fake_1.png')

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
output = cv2.resize(output,(512, 512))

# Display the output image
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()