######### Region Selector ###############################
"""
This script allows to collect raw points from an image.
The inputs are two mouse clicks one in the x,y position and
the second in w,h of a rectangle.
Once a rectangle is selected the user is asked to enter the type
and the Name:
Type can be 'Text' or 'CheckBox'
Name can be anything
"""

import cv2
import random

pathPdf = 'orijentiranipdfovi\\1.jpg'
pathPdfVeliki = "orijentiranipdfovi\\veliki\\1.jpg"
pathImg = "orijentirani\\1.jpg"
pathImgVeliki = "orijentirani\\veliki\\1.jpg"
scale = 0.5
circles = []
counter = 0
counter2 = 0

counterVeliki = 0
counterVeliki2 = 0

counterPdf = 0
counterPdf2 = 0

counterPdfVeliki = 0
counterPdfVeliki2 = 0
point1=[]
point2=[]
myPoints = [[(284, 150), (1534, 210), 'text', 'ime'], [(284, 208), (682, 268), 'broj', 'jmbag'], [(280, 272), (362, 328), 'broj', 'zadatak'], [(1410, 340), (1490, 394), 'broj', 'bodovi']]

myPointsVeliki = [[(244, 108), (1582, 170), 'text', 'ime'], [(250, 176), (678, 236), 'broj', 'jmbag'], [(244, 240), (334, 300), 'broj', 'zadatak'], [(1456, 308), (1542, 364), 'broj', 'bodovi']]


myPointsPdf = [[(282, 150), (1528, 196), 'path', 'ime'], [(282, 212), (682, 264), 'broj', 'jmbag'], [(284, 274), (360, 326), 'broj', 'zadatak'], [(1406, 327), (1488, 380), 'broj', 'bodovi']]

myPointsPdfVeliki = [[(244, 106), (1586, 164), 'text', 'ime'], [(246, 170), (676, 230), 'broj', 'jmbag'], [(246, 238), (332, 298), 'broj', 'zadatak'], [(1454, 302), (1544, 360), 'broj', 'bodovi']]

myColor=[]


myPoints = []
myPointsVeliki = []

myPointsPdf = []
myPointsPdfVeliki = []
def mousePoints(event,x,y,flags,params):
    global counter,point1,point2,counter2,circles,myColor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter==0:
            point1=int(x//scale),int(y//scale);
            counter +=1
            myColor = (random.randint(0,2)*200,random.randint(0,2)*200,random.randint(0,2)*200 )
        elif counter ==1:
            point2=int(x//scale),int(y//scale)
            type = input('Enter Type')
            name = input ('Enter Name ')
            myPoints.append([point1,point2,type,name])
            counter=0
        circles.append([x,y,myColor])
        counter2 += 1


img = cv2.imread(pathPdfVeliki)
img = cv2.resize(img, (0, 0), None, scale, scale)


while True:
    # To Display points
    for x,y,color in circles:
        cv2.circle(img,(x,y),3,color,cv2.FILLED)
    cv2.imshow("Original Image ", img)
    cv2.setMouseCallback("Original Image ", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(myPoints)
        break


