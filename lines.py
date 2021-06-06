import cv2
from PIL import Image
import matplotlib.pyplot as plt
def remove(image):
    #image = cv2.imread("segmenti2\\CIPsharp_20210326_110903_0035.jpgime8.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    vertical_kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    detected_lines1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detected_lines2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)


    cnts2 = cv2.findContours(detected_lines2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
    for c in cnts2:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    cnts1 = cv2.findContours(detected_lines1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
    for c in cnts1:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)


    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    return result

#    cv2.imshow('thresh', thresh)
#    cv2.imshow('detected_lines', detected_lines2)
#    cv2.imshow('image', image)
#    cv2.imshow('result', result)
#    cv2.waitKey()

#"CIPsharp_20210326_110903_0035.jpgime8.jpg"


def center(image):

    original = image.copy()

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        if h> 33 and w > 28:
            y = y + 8
            x = x +3
            w = 27
            h = 27

            ROI = original[y:y + h, x:x + w]


            return ROI
        if h < 11 and w < 11:
            continue

        if h >25 and w > 25:
            continue
        y = y - 2
        x = x - 3
        h = h + 7
        w = w + 8
        ROI = original[y:y + h, x:x + w]

        return ROI


#image1=cv2.imread("segmenti2\\CIPsharp_20210326_110903_0001.jpgjmbag8.jpg")
##image1=cv2.resize(image1,(28,28))
#cv2.imshow("",image1)
#cv2.waitKey(0)
#
###image1=cv2.imread("test2.jpg",cv2.IMREAD_GRAYSCALE)
###image1=cv2.resize(image1,(28,28))
##cv2.imshow("",image1)
##cv2.waitKey(0)
#image1=remove(image1)
#image1=cv2.resize(image1,(50,50))
#cv2.imshow("",image1)
#cv2.waitKey(0)
