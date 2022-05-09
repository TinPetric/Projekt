import os
import time
import cv2
import numpy as np
from boxdetect import config
from boxdetect.pipelines import get_boxes

def iterirajKrozGrupe(grupapodataka, vrstapodataka, nazivobrasca, brojzapisa, savedX1, savedX2, savedY1, savedY2):
    grupapodataka = sorted(grupapodataka, key=lambda x: x[0])
    if vrstapodataka == "bodovi":
        proslaxkoordinata = 842
    else:
        proslaxkoordinata = 4
    d = 0
    prvaXKoordinata = 0
    prvaYKoordinata = 0
    zadnjaXKoordinata = 0
    zadnjaYkoordinata = 0
    listaSegmenata = []
    for i in grupapodataka:

        y1 = i[1] +1
        y2 = y1 + i[3] - 1
        x1 = i[0] + 1
        x2 = x1 + i[2] - 2

        test = priblizeno[y1:y2, x1:x2]

        listaNeUlovljenih = []

        while x1 - proslaxkoordinata > 40:
            print("usao ako smo proskocili nes")
            x1KojiNijeUlovljen = x1 - 30
            x2KojiNijeUlovljen = x1
            kvadraticKojiNijeUlovljen = priblizeno[y1:y2, x1KojiNijeUlovljen:x2KojiNijeUlovljen]
            listaNeUlovljenih.append(kvadraticKojiNijeUlovljen)
            listaSegmenata.append(kvadraticKojiNijeUlovljen)
            brojzapisa = brojzapisa + 1
            x1 = x1 - 30

        if listaNeUlovljenih:

            for br in range(len(listaNeUlovljenih)):
                slika = listaNeUlovljenih.pop()


                print("upisan: " + str(nazivobrasca) + vrstapodataka + str(d) + ".jpg")
                d = d + 1
                print("nastavljamo dalje")


        x1 = i[0] + 1
        kvadratic = priblizeno[y1:y2, x1:x2]
        if brojzapisa == 0:
            prvaXKoordinata = x1
            prvaYKoordinata = y1

        if brojzapisa == 30:
            zadnjaXKoordinata = x2
            zadnjaYkoordinata = y2

        if brojzapisa == 1:
            zadnjaXKoordinata = x2
            zadnjaYkoordinata = y2
        ime = "test box detecta sa svim obrascima\\" + str(nazivobrasca) + vrstapodataka + str(d) + ".jpg"

        listaSegmenata.append(kvadratic)
        brojzapisa = brojzapisa + 1
        print("upisan: " + str(nazivobrasca) + vrstapodataka + str(d) + ".jpg")
        proslaxkoordinata = x1
        d = d + 1
    nestoupisano = False
    if vrstapodataka == "ime":
        if d == 30:
            kvadraticKojiNijeUlovljen = priblizeno[y1:y2, x2+ 2:x2 + 30]
            nestoupisano = True
    if vrstapodataka == "jmbag":
        if d == 9:
            kvadraticKojiNijeUlovljen = priblizeno[y1:y2, x2+ 2:x2 + 30]
            nestoupisano = True
    if vrstapodataka == "zadatak":
        if d == 1:
            kvadraticKojiNijeUlovljen = priblizeno[y1:y2, x2+ 2:x2 + 30]
            nestoupisano = True
    if vrstapodataka == "bodovi":
        if d == 1:
            kvadraticKojiNijeUlovljen = priblizeno[y1:y2, x2+ 2:x2 + 30]
            nestoupisano = True
    ime = "test box detecta sa svim obrascima\\" + str(nazivobrasca) + vrstapodataka + str(d) + ".jpg"
    uspjesnoSegmentirano = False
    if nestoupisano:
        brojzapisa = brojzapisa + 1
        listaSegmenata.append(kvadraticKojiNijeUlovljen)
        print("naknadno upisan zadnji")
    if vrstapodataka == 'ime':
        if brojzapisa == 31:
            uspjesnoSegmentirano =True
    if vrstapodataka == "jmbag":
        if brojzapisa == 10:
            uspjesnoSegmentirano = True
    if vrstapodataka == "zadatak":
        if brojzapisa == 2:
            uspjesnoSegmentirano = True
    if vrstapodataka == "bodovi":
        if brojzapisa == 2:
            uspjesnoSegmentirano = True

    if uspjesnoSegmentirano:
        d = 0
        for i in listaSegmenata:

            cv2.imshow("segment", i)
            ime = "test box detecta sa svim obrascima\\" + str(nazivobrasca) + vrstapodataka + str(d) + ".jpg"
            cv2.imwrite(ime, i)

            print(d)
            d =d +1
    else:
        snip = priblizeno[savedY1:savedY2, savedX1:savedX2]


        if vrstapodataka == "ime":


            snip = cv2.resize(snip, (930, 40))

            test_digits = np.hsplit(snip, 31)
        if vrstapodataka == "jmbag":

            snip =cv2.resize(snip, (300, 40))

            test_digits = np.hsplit(snip, 10)
        if vrstapodataka == "zadatak":

            snip =cv2.resize(snip, (60, 40))
            test_digits = np.hsplit(snip, 2)
        if vrstapodataka == "bodovi":

            snip =cv2.resize(snip, (60, 40))
            test_digits = np.hsplit(snip, 2)

        d=0
        for digit in test_digits:
            ime = "test box detecta sa svim obrascima\\" + str(nazivobrasca) + vrstapodataka + str(d) + ".jpg"
            digit = digit[3:37,2:28]
            digit = cv2.resize(digit, (30, 40))
            cv2.imwrite(ime, digit)
            d = d+1


    return brojzapisa

path = 'orijentirani2'
myPicList = os.listdir(path)
br = 1
imeX1= 17
imeX2 = 952
imeY1 = 21
imeY2 = 62

jmbagX1= 16
jmbagX2 = 317
jmbagY1 = 67
jmbagY2 = 107

zadatakX1= 16
zadatakX2 = 77
zadatakY1 = 113
zadatakY2 = 152

bodoviX1= 862
bodoviY1 = 158
bodoviX2 = 922

bodoviY2 = 199
for j,y in enumerate(myPicList):


    brojimena = 0
    brojjmbag = 0
    brojzadatak = 0
    brojbodovi = 0

    print(f'################## Extracting Data from Form {j} {y}  ##################')
    file_name = "zaboxdetect\\" + y

    img = cv2.imread(path + "\\" + y)
    priblizeno = img[90:330, 195:1170]


    cv2.imwrite(file_name, priblizeno)
    cfg = config.PipelinesConfig()

    cfg.width_range = (25, 31)
    cfg.height_range = (37,42)

    cfg.scaling_factors = [0.8]

    cfg.wh_ratio_range = (0.4, 1.2)

    cfg.group_size_range = (1, 100)

    cfg.dilation_iterations = 0


    rects, grouping_rects, image, output_image = get_boxes(
        file_name, cfg=cfg, plot=False)

    cv2.imwrite("probafolder\\slika.jpg", output_image)
    rects = rects.tolist()
    okrenuti = rects[::-1]


    print(len(okrenuti))

    imeiprezime = []
    jmbag = []
    zadatak = []
    bodovi = []
    for i in okrenuti:

        if i[1] > 17 and i[1] < 30:
            imeiprezime.append(i)
        elif i[1] >60 and i[1] < 75:
            jmbag.append(i)
        elif i[1] > 100 and i [1] < 120:
            zadatak.append(i)
        elif i[1] > 150 and i [1] < 170:
            bodovi.append(i)

    print("duljina" + str(len(imeiprezime)))

    if len(imeiprezime) == 31:
        slika = grouping_rects[0]
        imeX1  = slika[0]
        imeY1 = slika[1]
        imeX2 = slika[2] +  imeX1
        imeY2 = slika[3] + imeY1

    brojimena = iterirajKrozGrupe(imeiprezime, "ime", y, brojimena, imeX1, imeX2, imeY1, imeY2)

    if len(jmbag) == 10:
        slika = grouping_rects[1]
        jmbagX1  = slika[0]
        jmbagY1 = slika[1]
        jmbagX2 = slika[2] +  jmbagX1
        jmbagY2 = slika[3] + jmbagY1

    brojjmbag = brojjmbag + iterirajKrozGrupe(jmbag, "jmbag", y, brojjmbag, jmbagX1, jmbagX2, jmbagY1, jmbagY2)

    if len(zadatak) == 2:
        slika = grouping_rects[2]
        zadatakX1  = slika[0]
        zadatakY1 = slika[1]
        zadatakX2 = slika[2] + zadatakX1
        zadatakY2 = slika[3] + zadatakY1

    brojzadatak = brojzadatak + iterirajKrozGrupe(zadatak, "zadatak", y, brojzadatak, zadatakX1, zadatakX2, zadatakY1, zadatakY2)

    if len(bodovi) == 2:
        slika = grouping_rects[2]
        bodoviX1  = slika[0]
        bodoviY1 = slika[1]
        bodoviX2 = slika[2] + bodoviX1
        bodoviY2 = slika[3] + bodoviY1
    brojbodovi = brojbodovi + iterirajKrozGrupe(bodovi, "bodovi", y, brojbodovi, bodoviX1, bodoviX2, bodoviY1, bodoviY2)

    if not brojimena == 31:
        datotekaZaProvjeru = open("naknadnaprovjera.txt", "a")
        datotekaZaProvjeru.write("ime - " + y)
        datotekaZaProvjeru.write("\n")
        datotekaZaProvjeru.close()
    if not brojjmbag == 10:
        datotekaZaProvjeru = open("naknadnaprovjera.txt", "a")
        datotekaZaProvjeru.write("jmbag - " + y)
        datotekaZaProvjeru.write("\n")
        datotekaZaProvjeru.close()
    if not brojbodovi == 2:
        datotekaZaProvjeru = open("naknadnaprovjera.txt", "a")
        datotekaZaProvjeru.write("bodovi - " + y)
        datotekaZaProvjeru.write("\n")
        datotekaZaProvjeru.close()

    if not brojzadatak == 2:
        datotekaZaProvjeru = open("naknadnaprovjera.txt", "a")
        datotekaZaProvjeru.write("zadatak - " + y)
        datotekaZaProvjeru.write("\n")
        datotekaZaProvjeru.close()
    print("broj slicica imena : " + str(brojimena))
    print("broj slicica jmbaga : " + str(brojjmbag))
    print("broj slicica zadatak : " + str(brojzadatak))
    print("broj slicica bodovi : " + str(brojbodovi))



    br = br + 1