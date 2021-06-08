from math import floor

import numpy as np
import cv2
import matplotlib.pyplot as plt

def readButton(img):
    #img = cv2.imread("segmenti\\" + str(i) +"Bodovi.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)

    minDist = 100
    param1 = 10 #500
    param2 = 18 #200 #smaller value-> more false circles
    minRadius = 5
    maxRadius = 50 #10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    size = img.shape
    xvalue = size[1]/16
    print("xvalue: ",xvalue)
    result=0
    print("circles",circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            result=floor(i[0]/ xvalue)
            print("result",result)



    # Show result for testing:

    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    cv2.destroyAllWindows()
    return result