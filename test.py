import cv2
import numpy as np
import pytesseract
import os

# C:\Program Files\Tesseract-OCR

per = 25
pixelThreshold=500

roi=[[(214, 112), (1148, 152), 'text', 'ime'],
     [(216, 156), (510, 196), 'text', 'jmbag'],
     [(214, 204), (268, 240), 'text', 'zadatak'],
     [(1056, 254), (1114, 284), 'text', 'bodovi']]



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# images for machine learning
digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)


rows = np.vsplit(digits, 50)
cells = []
for row in rows:
    row_cells = np.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = np.array(cells, dtype=np.float32)


k = np.arange(10)

cells_labels = np.repeat(k, 250)


imgQ = cv2.imread('1.jpg', )
h,w,c = imgQ.shape
#imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'FilledForms'
myPicList = os.listdir(path)

for j,y in enumerate(myPicList):
    img = cv2.imread(path +"\\"+y, cv2.IMREAD_GRAYSCALE)

    print(f'################## Extracting Data from Form {j + 1}  ##################')

    for x,r in enumerate(roi):


        #ime i prezime
        imgCrop = img[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #print(f'{r[0][1]}:{r[1][1]}, {r[0][0]}:{r[1][0]}')

        cv2.imshow(str(x), imgCrop)
        if x==0:
            cv2.imshow("test", imgCrop)
            test_digits = np.hsplit(imgCrop, 31)
            cv2.imshow("test", test_digits[0])
            cv2.waitKey(0)
            for i in range(0,30):
                pom = []
                x2=x1+parts
                y2=h
                imgFinal=imgCrop[y1:y2, x1:x2]
                x1=x2
                pom.append(imgFinal)

                ime = []
                for d in pom:
                    d = d.flatten()
                    ime.append(d)
                ime = np.array(ime, dtype=np.float32)


        if x==1:

            h, w, c = imgCrop.shape
            parts = w // 10
            # print(parts)
            x1 = 0
            y1 = 0

            for i in range(0, 10):
                pom = []
                x2 = x1 + parts
                y2 = h
                imgFinal = imgCrop[y1:y2, x1:x2]
                x1 = x2
                pom.append(imgFinal)

            jmbag = []
            for d in pom:
                d = d.flatten()
                jmbag.append(d)
            jmbag = np.array(jmbag, dtype=np.float32)





        if x == 2:

            h, w, c = imgCrop.shape
            parts = w // 2
            # print(parts)
            x1 = 0
            y1 = 0
            for i in range(0, 2):
                pom = []
                x2 = x1 + parts
                y2 = h
                imgFinal = imgCrop[y1:y2, x1:x2]
                x1 = x2
                pom.append(imgFinal)

                zadatak = []
                for d in pom:
                    d = d.flatten()
                    zadatak.append(d)
                zadatak = np.array(zadatak, dtype=np.float32)


        if x==3:

            h, w, c = imgCrop.shape
            parts = w // 2
            # print(parts)
            x1 = 0
            y1 = 0

            for i in range(0, 2):
                pom = []
                x2 = x1 + parts
                y2 = h
                imgFinal = imgCrop[y1:y2, x1:x2]
                x1 = x2
                pom.append(imgFinal)

                bodovi = []
                for d in pom:
                    d = d.flatten()
                    bodovi.append(d)
                bodovi = np.array(bodovi, dtype=np.float32)


    cv2.waitKey(0)


cv2.waitKey(0)