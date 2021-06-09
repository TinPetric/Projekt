from boxdetect import config
from boxdetect.pipelines import get_boxes

import cv2

import numpy as np

import os
from PIL import Image

import lines
#import classify



import matplotlib.pyplot as plt

def detect(file_name):
   # file_name = 'orijentirani2\\CIPsharp_20210326_110903_0020.jpg'
    #file_name = '1.jpg'

    cfg = config.PipelinesConfig()

    # important to adjust these values to match the size of boxes on your image
    cfg.width_range = (23,37)
    cfg.height_range = (35,45)

    # the more scaling factors the more accurate the results but also it takes more time to processing
    # too small scaling factor may cause false positives
    # too big scaling factor will take a lot of processing time
    cfg.scaling_factors = [0.8]

    # w/h ratio range for boxes/rectangles filtering
    cfg.wh_ratio_range = (0.5, 1.7)

    # group_size_range starting from 2 will skip all the groups
    # with a single box detected inside (like checkboxes)
    cfg.group_size_range = (2, 40)

    # num of iterations when running dilation tranformation (to engance the image)
    cfg.dilation_iterations = 0



    rects, grouping_rects, image, output_image = get_boxes(file_name, cfg=cfg, plot=False)
    #print(rects)
    #print(grouping_rects)

    plt.figure(figsize=(20,20))
    plt.imshow(output_image)
    plt.show()

    return rects, grouping_rects, image, output_image



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



per = 25
pixelThreshold=500

#roi=[[(214, 112), (1148, 152), 'text', 'ime'],
#     [(216, 156), (510, 196), 'text', 'jmbag'],
#     [(214, 204), (268, 240), 'text', 'zadatak'],
#     [(1056, 254), (1114, 284), 'text', 'bodovi'],
#     [(536, 242), (1032, 306), ' BUTTONS', 'bodovi']]



#imgQ = cv2.imread('1.jpg', )
#h,w,c = imgQ.shape
##imgQ = cv2.resize(imgQ,(w//3,h//3))
#
#orb = cv2.ORB_create(1000)
#kp1, des1 = orb.detectAndCompute(imgQ,None)
##impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'orijentirani2'
#path="prvi_sken2"
myPicList = os.listdir(path)

#classify.train()
#new_model = tf.keras.models.load_model('epic_num_reader.h5')
for j,y in enumerate(myPicList):
    if "PDF" in y:
        continue
    file_path=path + "\\" + y
    rects, grouping_rects, image, output_image=detect(file_path)
    a=grouping_rects

    if len(grouping_rects)==4 and len(rects)==44:
        print(rects)
        roi = [[(int(a[0][0]),int(a[0][1])), (int(a[0][0]+int(a[0][2])),int(a[0][1])+ int(a[0][3])), 'text', 'ime'],
               [(int(a[1][0]),int(a[1][1])), (int(a[1][0]+int(a[1][2])),int(a[1][1])+ int(a[1][3])), 'text', 'jmbag'],
               [(int(a[2][0]),int(a[2][1])), (int(a[2][0]+int(a[2][2])),int(a[2][1])+ int(a[2][3])), 'text', 'zadatak'],
               [(int(a[3][0]),int(a[3][1])), (int(a[3][0]+int(a[3][2])),int(a[3][1])+ int(a[3][3])), 'text', 'bodovi']]
            ##   [(536, 242), (1032, 306), ' BUTTONS', 'bodovi']]
    else:
        roi = [[(214, 112), (1148, 152), 'text', 'ime'],
               [(216, 156), (510, 196), 'text', 'jmbag'],
               [(214, 204), (268, 240), 'text', 'zadatak'],
               [(1056, 254), (1114, 284), 'text', 'bodovi'],
               [(536, 242), (1032, 306), ' BUTTONS', 'bodovi']]
        #roi=[[(210, 112), (933, 43), 'text', 'ime'],
        #     [(210, 157), (302, 42), 'text', 'jmbag'],
        #     [(210, 203), (59, 42), 'text', 'zadatak'],
        #     [(1056, 250), (60, 42), 'text', 'bodovi']]
    #print(roi)

        #  [(210, 112, 933, 43), (210, 157, 302, 42), (210, 203, 59, 42), (1056, 250, 60, 42)]
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow("",img)
    #cv2.waitKey(0)
    for x,r in enumerate(roi):
        #print(x, r)
        #print(r[0][1],r[1][1], r[0][0],r[1][0])
        imgCrop = img[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow("",imgCrop)
        #cv2.waitKey(0)

        #segmentacija imena
        if x==0:
            #opet resizam sliku u ovom slucaju sliku imena i prezimena da mogu podjeliti sa hsplit
            imgCrop = cv2.resize(imgCrop, (899, 40))
            test_digits = np.hsplit(imgCrop, 31)

            for d, z in enumerate(test_digits):
                #tu resizam pojedine slicice da budu iste velicine, objasnjeno gore
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                #sprema u mapu segmenti da ih mozemo vidjeti lijepo
                ime = "segmenti3\\" + str(y) + "ime" + str(d) + ".jpg"

                cv2.imwrite(ime, test_digits[d])

            jmbag=[]
            for d,z in enumerate(test_digits):
                ime = "segmenti3\\" + str(y) + "ime" + str(d) + ".jpg"

                cv2.imwrite(ime, test_digits[d])
                im2 = cv2.imread(ime)
                test_digits[d] = lines.remove(im2)

                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
               # crop_img = test_digits[d][2:24, 3:25]
                #crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(test_digits[d])

                _,sharpen= cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY)

                cv2.imwrite(ime, sharpen)

        #segmentacija za jmbag
        if x==1:
            #sve isto kao i kod imena
            imgCrop = cv2.resize(imgCrop, (300, 40))
            test_digits = np.hsplit(imgCrop, 10)

            for d,z in enumerate(test_digits):
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                ime = "segmenti3\\" + str(y) + "jmbag" + str(d) + ".jpg"
                #test_digits[d]=cv2.resize(test_digits[d],(28,28))
                cv2.imwrite(ime, test_digits[d])

            jmbag=[]
            for d,z in enumerate(test_digits):
                ime = "segmenti3\\" + str(y) + "jmbag" + str(d) + ".jpg"

                cv2.imwrite(ime, test_digits[d])
                im2 = cv2.imread(ime)
                test_digits[d] = lines.remove(im2)

                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
               # crop_img = test_digits[d][2:24, 3:25]
                #crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(test_digits[d])

                _,sharpen= cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY)

                cv2.imwrite(ime, sharpen)

                im2=cv2.imread(ime)[:,:,0]
                im2 = np.invert(np.array([im2]))

                im2 = im2 / 255
                #im2 = tf.keras.utils.normalize(im2, axis=-1)
                #predictions = new_model.predict(im2)

                #print(predictions)
                #print('prediction -> ', np.argmax(predictions))
                #jmbag.append(np.argmax(predictions))
                #plt.imshow(im2[0], cmap=plt.cm.binary)
                #plt.show()

            #print(jmbag)


        if x == 2:
           # cv2.imshow("1",imgCrop)
            imgCrop = cv2.resize(imgCrop, (60, 40))
            test_digits = np.hsplit(imgCrop, 2)

            for d, z in enumerate(test_digits):
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                ime = "segmenti3\\" + str(y) + "zadatak" + str(d) + ".jpg"
                cv2.imwrite(ime, test_digits[d])

            jmbag=[]
            for d,z in enumerate(test_digits):
                ime = "segmenti3\\" + str(y) + "zadatak" + str(d) + ".jpg"

                cv2.imwrite(ime, test_digits[d])
                im2 = cv2.imread(ime)
                test_digits[d] = lines.remove(im2)

                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
               # crop_img = test_digits[d][2:24, 3:25]
                #crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(test_digits[d])

                _,sharpen= cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY)

                cv2.imwrite(ime, sharpen)


        if x==3:
            imgCrop = cv2.resize(imgCrop, (60, 40))
            test_digits = np.hsplit(imgCrop, 2)
            for d, z in enumerate(test_digits):
                test_digits[d] = cv2.resize(test_digits[d], (35, 40))
                ime = "segmenti3\\" + str(y) + "bodovi" + str(d) + ".jpg"
                cv2.imwrite(ime, test_digits[d])

            jmbag=[]
            for d,z in enumerate(test_digits):
                ime = "segmenti3\\" + str(y) + "bodovi" + str(d) + ".jpg"

                cv2.imwrite(ime, test_digits[d])
                im2 = cv2.imread(ime)
                test_digits[d] = lines.remove(im2)

                test_digits[d] = cv2.resize(test_digits[d], (28, 28))
               # crop_img = test_digits[d][2:24, 3:25]
                #crop_img = cv2.resize(crop_img, (28, 28))
                sharpen = unsharp_mask(test_digits[d])

                _,sharpen= cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY)

                cv2.imwrite(ime, sharpen)



        if x==4:
            ime = "segmenti3\\" + str(y) + "Bodovi.jpg"
            cv2.imwrite(ime, imgCrop)

        cv2.waitKey(0)



cv2.waitKey(0)