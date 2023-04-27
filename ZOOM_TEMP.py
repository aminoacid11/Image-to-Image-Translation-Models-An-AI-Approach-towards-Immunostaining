import cv2
import numpy as np
from skimage.metrics import structural_similarity

def show_diff_image(img_num):
    img1 = cv2.imread("data/membrane/IMAGES/r/real_{}.png".format(img_num))
    img2 = cv2.imread("data/membrane/IMAGES/f/fake_{}.png".format(img_num))

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(g1, g2, full=True)
    diff = (diff*255).astype("uint8")

    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if cv2.contourArea(c) > 80]

    filled_img1 = img1.copy()
    filled_img2 = img2.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 4)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 4)
        cv2.drawContours(filled_img1, [c], 0, (0, 255, 0), -1)
        cv2.drawContours(filled_img2, [c], 0, (0, 255, 0), -1)
        cv2.putText(img2, "Similarity: " + str(score), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    x = np.zeros((2048, 10, 3), np.uint8)
    result = np.hstack((img1, x, img2))
    result2 = np.hstack((filled_img1, x, filled_img2))

    cv2.imshow("Differences", cv2.resize(result, None, fx=0.35, fy=0.35))
    cv2.imshow('Filled img2', cv2.resize(result2, None, fx=0.35, fy=0.35))

    while True:
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

        if key in [ord('1'), ord('2')]:
            index = 0 if key == ord('1') else 1
            img = img1 if index == 0 else img2

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cropped_img = img[y:y+h, x:x+w]

                # Zoom into the selected area
                zoom = cv2.resize(cropped_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Zoomed Image", zoom)

show_diff_image(1)