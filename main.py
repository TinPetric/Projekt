import cv2
import numpy as np
import pytesseract
import os

# C:\Program Files\Tesseract-OCR

per = 25
pixelThreshold=500

roi=[[(250, 110), (1592, 164), ' Letters', 'Name'],
     [(250, 176), (680, 232), ' Digits', 'JMBAG'],
     [(248, 238), (334, 298), ' Digits', 'Number'],
     [(718, 304), (1422, 386), ' Buttons', 'Points'],
     [(1464, 306), (1548, 362), ' Digits', 'Points']]



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('form.png')
h,w,c = imgQ.shape
#imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'FilledForms'
myPicList = os.listdir(path)
for j,y in enumerate(myPicList):
    img = cv2.imread(path +"/"+y)
    img = cv2.resize(img, (w, h))

    print(f'################## Extracting Data from Form {j}  ##################')

    for x,r in enumerate(roi):

        imgCrop = img[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #print(f'{r[0][1]}:{r[1][1]}, {r[0][0]}:{r[1][0]}')
        cv2.imshow(str(x), imgCrop)

        if x==0:
            h,w,c=imgCrop.shape
            parts=w//30-1
            #print(parts)
            x1=0
            y1=0

            for i in range(0,30):
                x2=x1+parts
                y2=h
                imgFinal=imgCrop[y1:y2, x1:x2]
                x1=x2
                cv2.imshow(str(i+5),imgFinal)



cv2.waitKey(0)