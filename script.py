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
myPoints = []
myPointsVeliki = []
myPointsPdf = []
myPointsPdfVeliki = []
myColor=[]
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

def mousePointsVeliki(event,x,y,flags,params):
    global counterVeliki,point1,point2,counterVeliki2,circles,myColor
    counterVeliki = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        if counterVeliki==0:
            point1=int(x//scale),int(y//scale);
            counterVeliki +=1
            myColor = (random.randint(0,2)*200,random.randint(0,2)*200,random.randint(0,2)*200 )
        elif counterVeliki ==1:
            point2=int(x//scale),int(y//scale)
            type = input('Enter Type')
            name = input ('Enter Name ')
            myPointsVeliki.append([point1,point2,type,name])
            counterVeliki=0
        circles.append([x,y,myColor])
        counterVeliki2 += 1

def mousePointsPdf(event,x,y,flags,params):
    global counterPdf,point1,point2,counterPdf2,circles,myColor
    counterPdf = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        if counterPdf==0:
            point1=int(x//scale),int(y//scale);
            counterPdf +=1
            myColor = (random.randint(0,2)*200,random.randint(0,2)*200,random.randint(0,2)*200 )
        elif counterPdf ==1:
            point2=int(x//scale),int(y//scale)
            type = input('Enter Type')
            name = input ('Enter Name ')
            myPointsPdf.append([point1,point2,type,name])
            counterPdf=0
        circles.append([x,y,myColor])
        counterPdf2 += 1

def mousePointsPdfVeliki(event,x,y,flags,params):
    global counterPdfVeliki,point1,point2,counterPdfVeliki2,circles,myColor
    counterPdfVeliki = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        if counterPdfVeliki==0:
            point1=int(x//scale),int(y//scale);
            counterPdfVeliki +=1
            myColor = (random.randint(0,2)*200,random.randint(0,2)*200,random.randint(0,2)*200 )
        elif counterPdfVeliki ==1:
            point2=int(x//scale),int(y//scale)
            type = input('Enter Type')
            name = input ('Enter Name ')
            myPointsPdfVeliki.append([point1,point2,type,name])
            counterPdfVeliki=0
        circles.append([x,y,myColor])
        counterPdfVeliki2 += 1
img = cv2.imread(pathImg)
img = cv2.resize(img, (0, 0), None, scale, scale)

imgVeliki = cv2.imread(pathImgVeliki)
imgVeliki = cv2.resize(imgVeliki, (0, 0), None, scale, scale)

imgPdf = cv2.imread(pathPdf)
imgPdf = cv2.resize(imgPdf, (0, 0), None, scale, scale)

imgPdfVeliki = cv2.imread(pathPdfVeliki)
imgPdfVeliki = cv2.resize(imgPdfVeliki, (0, 0), None, scale, scale)
while True:
    # To Display points
    for x,y,color in circles:
        cv2.circle(img,(x,y),3,color,cv2.FILLED)
    cv2.imshow("Original Image ", img)
    cv2.setMouseCallback("Original Image ", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(myPoints)
        break


scale = 0.5
circles = []
counter = 0
counter2 = 0
point1=[]
point2=[]
while True:
    for x, y, color in circles:
        cv2.circle(imgVeliki, (x, y), 3, color, cv2.FILLED)
    cv2.imshow("Original Image ", imgVeliki)
    cv2.setMouseCallback("Original Image ", mousePointsVeliki)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(myPointsVeliki)
        break


scale = 0.5
circles = []
counter = 0
counter2 = 0
point1=[]
point2=[]

while True:
    for x, y, color in circles:
        cv2.circle(imgPdf, (x, y), 3, color, cv2.FILLED)
    cv2.imshow("Original Image ", imgPdf)
    cv2.setMouseCallback("Original Image ", mousePointsPdf)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(mousePointsPdf())
        break

scale = 0.5
circles = []
counter = 0
counter2 = 0
point1=[]
point2=[]

while True:
    for x, y, color in circles:
        cv2.circle(imgPdfVeliki, (x, y), 3, color, cv2.FILLED)
    cv2.imshow("Original Image ", imgPdfVeliki)
    cv2.setMouseCallback("Original Image ", mousePointsPdfVeliki)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(mousePointsPdfVeliki())
        break