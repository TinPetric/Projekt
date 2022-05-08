import cv2
from boxdetect import config
from boxdetect.pipelines import get_boxes

def iterirajKrozGrupe(grupapodataka):
    grupapodataka = sorted(grupapodataka, key=lambda x: x[0])
    proslaxkoordinata = grupapodataka[0][0]
    for i in grupapodataka:

        y1 = i[1]
        y2 = y1 + i[3]
        x1 = i[0]
        x2 = x1 + i[2]

        if x1 - proslaxkoordinata > 40:
            print("usao ako smo proskocili nes")
            x1KojiNijeUlovljen = x1 - 30
            x2KojiNijeUlovljen = x1
            kvadraticKojiNijeUlovljen = priblizeno[y1:y2, x1KojiNijeUlovljen:x2KojiNijeUlovljen]
            cv2.imshow("kvadratickoji nije ulovljen", kvadraticKojiNijeUlovljen)
            cv2.waitKey(0)
            print("nastavljamo dalje")

        kvadratic = priblizeno[y1:y2, x1:x2]

        cv2.imshow("kvadratic", kvadratic)
        cv2.waitKey(0)
        proslaxkoordinata = x1


file_name = "zaboxdetect\\1.jpg"

slika = cv2.imread("FilledForms\\1.jpg")
priblizeno = slika[90:330, 195:1170]

cv2.imwrite("zaboxdetect\\1.jpg", priblizeno)
cfg = config.PipelinesConfig()

cfg.width_range = (25, 31)
cfg.height_range = (37,42)

cfg.scaling_factors = [0.8]

cfg.wh_ratio_range = (0.4, 1.2)

cfg.group_size_range = (1, 100)

cfg.dilation_iterations = 0


rects, grouping_rects, image, output_image = get_boxes(
    file_name, cfg=cfg, plot=False)

print("koordinate : ")
print(grouping_rects)

print("pojedini:")
print(rects)

print("okrenuti:")
rects = rects.tolist()
okrenuti = rects[::-1]
print(okrenuti)
slika = cv2.imread("FilledForms\\1.jpg")
print(len(okrenuti))
cv2.imshow("ime", output_image)
cv2.waitKey(0)
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

iterirajKrozGrupe(imeiprezime)

iterirajKrozGrupe(jmbag)

iterirajKrozGrupe(zadatak)

iterirajKrozGrupe(bodovi)