from __future__ import print_function
from pdf2image import convert_from_path
import glob
import os
from PIL import Image
import argparse
import cv2
import numpy as np

# Postaviti sve slike i pdfove u folder prvi_scan, pokrenuti renaming.py
# nakon toga preimenovane slike odvojiti iz foldera prvi_scan i pdfovi u
# foldere prvi_scan/veliki i pdfovi/veliki, nakon toga pokrenuti
# reorientation.py

poppler_path = "C:\Program Files\poppler-21.03.0\Library\\bin"
imgoviZaPremjestiti = [19, 20,21,22,23,24,25,26,27,28,29,30,31,32,
                       33,34,35,36,37,38,39,40,48,49,51,52,53,54,
                       78,79,80,83,84,87,88,89,90,93,94,99,100,103,104,
                       105,106,107,108,109,110,111,112,113,114,115,116,
                       117,118,119,120,121,124,125,151,152,153,154,155,
                       156,157,159,160,165,166,167,168,169,170,171,172,
                       173,174,175,176,177,178,179,180,181,182,183,184,185,186]

pdfoviZaPremjestiti = [1,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
                       34,35,42,43,44,45,50,51,52,59,60,67,79,80,82,83,85,86,87,
                       88,91,92,96,97,100,101,102,104,105,106,107,108,109,110,111,112,
                       113,114,117,118,119,120,121,122,123,124,125,126,127,128,129,130,133,134,
                       155,156,157,158,159,161,169,170,171,172,173,174,175,176,
                       177,178,179,180,181,182,183,184,185,186,187]

velikeOdvojiti = True
pdfoveOdvojiti = True

# Ako pdfovi nisu odvojeni postaviti zastavicu na True
if pdfoveOdvojiti:
    iterator = 1
    iteratorPdf = 1
    for filename in os.listdir("prvi_scan\\"):

        if filename == "veliki" or filename == "form.jpg":
            continue
        zaIspis = "Renaming " + filename + " to "
        if not filename.endswith(".pdf"):
            slika = Image.open("prvi_scan\\" + filename)
            velicina = slika.size
            slika.close()
        if filename.endswith(".pdf"):
            path = "prvi_scan\\" + filename

            image = convert_from_path("prvi_scan\\" + filename, 500,
                                      poppler_path=r'C:\Program Files\poppler-21.03.0\Library\bin')

            image[0].save("pdfovi\\" + str(iteratorPdf) + ".jpg", 'JPEG')
            img = Image.open("pdfovi\\" + str(iteratorPdf) + ".jpg")
            resized = img.resize(velicina)
            resized.save("pdfovi\\" + str(iteratorPdf) + ".jpg")
            os.remove("prvi_scan\\" + filename)
            zaIspis = zaIspis + str(iteratorPdf) + ".jpg" " in folder pdfovi"
            iteratorPdf = iteratorPdf + 1

        else:

            os.rename("prvi_scan\\" + filename, "prvi_scan\\" + str(iterator) + ".jpg")
            zaIspis = zaIspis + str(iterator) + ".jpg"
            iterator = iterator + 1
        print(zaIspis)

print("Renaming finished")

if velikeOdvojiti:
    for i in imgoviZaPremjestiti:
        imeSlike = str(i) + ".jpg"
        img = cv2.imread("prvi_scan\\" + imeSlike)
        cv2.imwrite("prvi_scan\\veliki\\" + imeSlike, img)
        os.remove("prvi_scan\\" + imeSlike)
        zaIspis = print("Transfering " + imeSlike + " from folder prvi_scan to folder prvi_scan\ veliki")
        print(zaIspis)
    for i in pdfoviZaPremjestiti:
        imeSlike = str(i) + ".jpg"
        img = cv2.imread("pdfovi\\" + imeSlike)
        cv2.imwrite("pdfovi\\veliki\\" + imeSlike, img)
        os.remove("pdfovi\\" + imeSlike)
        zaIspis = print("Transfering " + imeSlike + " from folder pdfovi to folder pdfovi\ veliki")
        print(zaIspis)