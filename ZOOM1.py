import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity

img_num = 1
# Load images
img1 = cv2.imread("data/membrane/IMAGES/r/real_{}.png".format(img_num))
img2 = cv2.imread("data/membrane/IMAGES/f/fake_{}.png".format(img_num))

# Resize
img1 = cv2.resize(img1, (512,512))
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Grayscale
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find the difference between two images
# Compute the mean structural similarity index
(similar, diff) = structural_similarity(g1,g2,full=True)

# diff is in range [0,1]. Convert it in range [0,255]
diff = (diff*255).astype("uint8")

# Apply threshold
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Loop over each contour
for contour in contours:
    if cv2.contourArea(contour) > 100:
        # Calculate bounding box
        x,y,w,h = cv2.boundingRect(contour)
        # Draw rectangle - bounding box
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img2, "Similarity: " + str(similar), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
        # Zoom into the rectangle
        zoomed_in_img1 = img1[y:y+h, x:x+w]
        zoomed_in_img2 = img2[y:y+h, x:x+w]
        # Show the zoomed in image
        cv2.imshow("Zoomed in image 1", zoomed_in_img1)
        cv2.imshow("Zoomed in image 2", zoomed_in_img2)
        cv2.waitKey(0)

# Show final images with differences
x = np.zeros((512,10,3), np.uint8)
result = np.hstack((img1, x, img2))
cv2.imshow("Differences", result)

cv2.waitKey(0)
cv2.destroyAllWindows()