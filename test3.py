from math import floor, ceil

import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



def readButton(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)
    minDist = 250
    param1 = 20 #500
    param2 = 10 #200 #smaller value-> more false circles
    minRadius = 1
    maxRadius = 25 #10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 2, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    size = img.shape
    xvalue= size[1]/16
    result=None
    print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            value = i[0]/ xvalue
            if value - floor(value) > 0.8:
                result=ceil(value)
            else:
                result=floor(i[0]/ xvalue)
    return result

    #cv2.imwrite("test1.jpg", img)


    #root = tk.Tk()

    #img = ImageTk.PhotoImage(Image.open("test1.jpg"))
    #label = tk.Label(root, image=img).pack()

    #root.after(1000, lambda: root.destroy())
    #root.mainloop()
    #os.remove("test1.jpg")
    # Show result for testing:


    #cv2.destroyAllWindows()
