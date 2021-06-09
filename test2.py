import os
from math import floor

import numpy as np
import cv2
import matplotlib.pyplot as plt


def unsharp_mask(image, kernel_size=(5, 5), sigma=5.0, amount=8.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)


    return sharpened

list = os.listdir("segmentiTrain")
path = 'segmentiTrain\\'
for i in list:
    if "Bodovi" in i:
        img = cv2.imread(path+i)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)

        minDist = 100
        param1 = 150 #500
        param2 = 10 #200 #smaller value-> more false circles
        minRadius = 5
        maxRadius = 50 #10

        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        size = img.shape
        xvalue = size[1]/16
        #print(xvalue)
        result=None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                result=floor(i[0]/ xvalue)
        print(result)

        # Show result for testing:

        #plt.imshow(img, cmap=plt.cm.binary)
        #plt.show()
        #cv2.destroyAllWindows()
