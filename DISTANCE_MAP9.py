import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

num = 5
fig, axs = plt.subplots(nrows=num, ncols=2, figsize=(10, 50))
for i in range(1, num+1):
    # Read the real and synthetic images
    real_img  = cv2.imread(f"data/membrane/IMAGES/r/real_{i}.png")
    synth_img  = cv2.imread(f"data/membrane/IMAGES/f/fake_{i}.png")

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
    axs[i-1, 0].imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
    axs[i-1, 1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    axs[i-1, 0].axis('off')
    axs[i-1, 1].axis('off')
    
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()