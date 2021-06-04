import cv2
import numpy as np
#import pytesseract
import os
from PIL import Image
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
    # view result


    # save reentered image

def add_margin(pil_img):
    width, height = pil_img.size
    left =round( (30 - width) / 2)
    right = left
    top = round((30 - height) / 2)
    bottom = top
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new('RGBA', (30, 30), (255,255,255))
    result.paste(pil_img, (left, top))
    result.convert('LA')
    return result

per = 25
pixelThreshold=500

roi=[[(284, 150), (1534, 210), 'text', 'ime'], [(284, 208), (682, 268), 'broj', 'jmbag'], [(280, 272), (362, 328), 'broj', 'zadatak'], [(1410, 340), (1490, 394), 'broj', 'bodovi']]
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])


#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# images for machine learning
#znamenke koje se narezu jedna po jedna i daju se algoritmu da uci


#prvo moram resizate sliku na te pixele jer inace fja vsplit ne radi, velicina slike mora biti visekratnik broja segmenata
#znaci rastavljam sliku na 10 djelova horizontalno i 16 djelova vertikalno



# tu se slaze da algoritam zna koja slicica iz ove prve slike za vjezbanje znaci koji broj,
# znaci tu se algoritmu govori da je prvih 16 slicica 0, drugih 16 slicica 1 itd


imgQ = cv2.imread("prvi_scan\\1.jpg")
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

            imeFoldera = "segmenti\\ime\\" + str(j + 1)
            if not os.path.exists(imeFoldera):
                os.mkdir(imeFoldera)
            for d, z in enumerate(test_digits):

                #tu resizam pojedine slicice da budu iste velicine, objasnjeno gore
                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
                #sprema u mapu segmenti da ih mozemo vidjeti lijepo
                ime = imeFoldera + "\\" +  str(d+1) + ".jpg"
                img1 = test_digits[d]
                crop_img = img1[2:24, 3:25]
                crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(crop_img)
                cv2.imwrite(ime, sharpen)
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
            imeFoldera = "segmenti\\jmbag\\" + str(j + 1)
            if not os.path.exists(imeFoldera):
                os.mkdir(imeFoldera)
            for d,z in enumerate(test_digits):
                # test_digits[d] = center(test_digits[d])
                # cv2.imwrite("pom.png", test_digits[d])
                # pilImg = Image.open("pom.png")
                # pilImg = add_margin(pilImg)
                # pilImg.save("pom.png")
                # test_digits[d] = cv2.imread("pom.png",cv2.IMREAD_GRAYSCALE)
                # os.remove("pom.png")
                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
                ime = imeFoldera + "\\" +  str(d+1) + ".jpg"
                img1 = test_digits[d]
                crop_img = img1[2:24, 3:25]
                crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(crop_img)
                cv2.imwrite(ime, sharpen)

            test_cells = []

            for d in test_digits:
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)




        #segmentacija za zadatak, sve isto kao i kod jmbaga
        if x == 2:
            imgCrop = cv2.resize(imgCrop, (60, 40))
            test_digits = np.hsplit(imgCrop, 2)
            imeFoldera = "segmenti\\zadatak\\" + str(j + 1)
            if not os.path.exists(imeFoldera):
                os.mkdir(imeFoldera)
            for d, z in enumerate(test_digits):
                # test_digits[d] = center(test_digits[d])
                # cv2.imwrite("pom.png", test_digits[d])
                # pilImg = Image.open("pom.png")
                # pilImg = add_margin(pilImg)
                # pilImg.save("pom.png")
                # test_digits[d] = cv2.imread("pom.png",cv2.IMREAD_GRAYSCALE)
                # os.remove("pom.png")
                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
                ime = imeFoldera + "\\" +  str(d+1) + ".jpg"
                img1 = test_digits[d]
                crop_img = img1[2:24, 3:25]
                crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(crop_img)
                cv2.imwrite(ime, sharpen)
            test_cells = []

            for d in test_digits:
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)



        #segmentacia za bodove, nisam uveo jos segmentaciju za popunjeni kruzic za broj bodova
        #ovdje je isto sve isto kao i kod jmbaga i zadatka
        if x==3:
            imgCrop = cv2.resize(imgCrop, (60, 40))
            test_digits = np.hsplit(imgCrop, 2)
            imeFoldera = "segmenti\\bodovi\\" + str(j + 1)
            if not os.path.exists(imeFoldera):
                os.mkdir(imeFoldera)
            for d, z in enumerate(test_digits):
                #test_digits[d] = center(test_digits[d])
                #cv2.imwrite("pom.png", test_digits[d])
                #pilImg = Image.open("pom.png")
                #pilImg = add_margin(pilImg)
                #pilImg.save("pom.png")
                #test_digits[d] = cv2.imread("pom.png",cv2.IMREAD_GRAYSCALE)
                #os.remove("pom.png")
                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
                ime = imeFoldera + "\\" +  str(d +1) + ".jpg"
                img1 = test_digits[d]
                crop_img = img1[2:24, 3:25]
                crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(crop_img)
                cv2.imwrite(ime, sharpen)
            test_cells = []

            for d in test_digits:
                d = d.flatten()
                test_cells.append(d)
            test_cells = np.array(test_cells, dtype=np.float32)



        if x==4:
            ime = "segmenti\\" + str(j) + "Bodovi.jpg"
            cv2.imwrite(ime, imgCrop)





cv2.waitKey(0)