import cv2
import numpy as np
#import pytesseract
import os
from PIL import Image

per = 25
pixelThreshold=500

roi=[[(214, 112), (1148, 152), 'text', 'ime'],
     [(216, 156), (510, 196), 'text', 'jmbag'],
     [(214, 204), (268, 240), 'text', 'zadatak'],
     [(1056, 254), (1114, 284), 'text', 'bodovi'],
     [(536, 242), (1032, 306), ' BUTTONS', 'bodovi']]



#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# images for machine learning
#znamenke koje se narezu jedna po jedna i daju se algoritmu da uci
digits = cv2.imread("digitstest.jpg",cv2.IMREAD_GRAYSCALE)

#prvo moram resizate sliku na te pixele jer inace fja vsplit ne radi, velicina slike mora biti visekratnik broja segmenata
#znaci rastavljam sliku na 10 djelova horizontalno i 16 djelova vertikalno
digits = cv2.resize(digits , (800,470))
rows = np.vsplit(digits, 10)
cells = []
for row in rows:
    #spremam pojedinu slicicu u ovo row_cells
    row_cells = np.hsplit(row, 16)
    for i,x in enumerate(row_cells):
        #sada moram resizati pojedinu slicicu ovdje i kasnije svaku nasu pojedinu slicicu sa obrazca
        #jer algoritam za machine learning mora imati slike istih dimenzija
        row_cells[i] = cv2.resize(row_cells[i], (35, 40))

    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)

cells = np.array(cells, dtype=np.float32)

# tu se slaze da algoritam zna koja slicica iz ove prve slike za vjezbanje znaci koji broj,
# znaci tu se algoritmu govori da je prvih 16 slicica 0, drugih 16 slicica 1 itd
k = np.arange(10)

cells_labels = np.repeat(k, 16)


imgQ = cv2.imread('1.jpg', )
h,w,c = imgQ.shape
#imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'FilledForms'
myPicList = os.listdir(path)
for j,y in enumerate(myPicList):

    img = cv2.imread(path + "\\" + y, cv2.IMREAD_GRAYSCALE)

    print(f'################## Extracting Data from Form {j} {y}  ##################')

    for x,r in enumerate(roi):

        imgCrop = img[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        #segmentacija imena
        if x==0:
            #opet resizam sliku u ovom slucaju sliku imena i prezimena da mogu podjeliti sa hsplit
            imgCrop = cv2.resize(imgCrop, (899, 40))
            test_digits = np.hsplit(imgCrop, 31)

            for d, z in enumerate(test_digits):
                #tu resizam pojedine slicice da budu iste velicine, objasnjeno gore
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                #sprema u mapu segmenti da ih mozemo vidjeti lijepo
                ime = "segmenti\\" + str(j) + "ime" + str(d) + ".jpg"
                cv2.imwrite(ime, test_digits[d])
            test_cells = []

            for d in test_digits:
                #tu provodim dio bitan za algoritam, nez tocno zasto al treba nad slicicama biti fja flatten
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)

        #segmentacija za jmbag
        if x==1:
            #sve isto kao i kod imena
            imgCrop = cv2.resize(imgCrop, (300, 40))
            test_digits = np.hsplit(imgCrop, 10)

            for d,z in enumerate(test_digits):
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                ime = "segmenti\\" + str(j) + "jmbag" + str(d) + ".jpg"
                cv2.imwrite(ime, test_digits[d])

            test_cells = []

            for d in test_digits:
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)


            #ovo je algoritam za machine learning
            #prvo se kreira
            knn = cv2.ml.KNearest_create()
            #zatim trenira
            #u njega mu saljem one slicice sa prve opisane slike, na pocetku programa - cells-  da on nad njima trenira
            #i cells_labels - to je ono sto sam opisao 16 puta 0, 16 puta 1 itd. da algoritam zna da su prvih 16 slika 0
            #drugih 16 jedinice
            knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
            ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)

            #ovdje se ispisuje rezultat, trenutno je zakomentiran
            #print(result)

        #segmentacija za zadatak, sve isto kao i kod jmbaga
        if x == 2:
            imgCrop = cv2.resize(imgCrop, (60, 40))
            test_digits = np.hsplit(imgCrop, 2)

            for d, z in enumerate(test_digits):
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                ime = "segmenti\\" + str(j) + "zadatak" + str(d) + ".jpg"
                cv2.imwrite(ime, test_digits[d])
            test_cells = []

            for d in test_digits:
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)

                # KNN
            knn = cv2.ml.KNearest_create()
            knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
            ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)

            print(result)

        #segmentacia za bodove, nisam uveo jos segmentaciju za popunjeni kruzic za broj bodova
        #ovdje je isto sve isto kao i kod jmbaga i zadatka
        if x==3:
            imgCrop = cv2.resize(imgCrop, (60, 40))
            test_digits = np.hsplit(imgCrop, 2)

            for d, z in enumerate(test_digits):
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                ime = "segmenti\\" + str(j) + "bodovi" + str(d) + ".jpg"
                cv2.imwrite(ime, test_digits[d])
            test_cells = []

            for d in test_digits:
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)

            # KNN
            knn = cv2.ml.KNearest_create()
            knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
            ret, result, neighbours, dist = knn.findNearest(test_cells, k=2)

            print(result)

        if x==4:
            ime = "segmenti\\" + str(j) + "Bodovi.jpg"
            cv2.imwrite(ime, imgCrop)

        cv2.waitKey(0)



cv2.waitKey(0)