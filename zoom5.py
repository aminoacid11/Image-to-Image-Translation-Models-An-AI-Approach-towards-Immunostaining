import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity

img_num = 3
# Load images
img1 = cv2.imread("data/membrane/IMAGES/r/real_{}.png".format(img_num))
img2 = cv2.imread("data/membrane/IMAGES/f/fake_{}.png".format(img_num))

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(g1,g2,full=True)
print("Image Similarity: {:.4f}%".format(score * 100))
diff = (diff*255).astype("uint8")

_, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY_INV)

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = [c for c in contours if cv2.contourArea(c) > 80]

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,255), 4)
    cv2.rectangle(img2, (x,y), (x+w, y+h), (0,0,255), 4)
    cv2.putText(img2, "Similarity: " + str(score), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    # # Zoom in
    zoom1 = img1[y:y+h, x:x+w]
    zoom2 = img2[y:y+h, x:x+w]
    zoom1 = cv2.resize(zoom1, (w*3,h*3))
    zoom2 = cv2.resize(zoom2, (w*3,h*3))
    cv2.imshow("Zoomed real", zoom1)
    cv2.imshow("Zoomed fake", zoom2)
    cv2.moveWindow("Zoomed real", 500, 100)
    cv2.moveWindow("Zoomed fake", 800, 100)
    cv2.waitKey(0)

# Show final images with differences
x = np.zeros((2048,10,3), np.uint8)
result = np.hstack((img1, x, img2))
cv2.imshow("Differences", cv2.resize(result, None, fx=0.35, fy=0.35))

cv2.waitKey(0)
cv2.destroyAllWindows()