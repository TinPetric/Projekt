import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

test = []
pathJmbag = "segmenti\\jmbag\\" + str(1)
list = os.listdir(pathJmbag)
for j in range(len(list)):
    ime = pathJmbag + "\\" + str(j + 1) + ".jpg"
    # load the image
    img = cv2.imread(ime)[:, :, 0]
    img = np.invert(np.array([img]))
    test.append(img)
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()


testy = [0 , 0 , 3, 6,5,1,2,3,4,5]
test = np.array(test)
testy = np.array(testy)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train =tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)



plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
img = cv2.imread("segmenti\\jmbag\\1\\10.jpg")


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(test, testy, epochs=10

          )




ime =[]
jmbag = []
zadatak = []
bodovi = []


model.save('digits.model')

list = os.listdir("segmenti\\zadatak\\")

for i in range(len(list)):

    # jmbag

    pathJmbag = "segmenti\\jmbag\\" + str(i + 1)

    jmbagList = os.listdir(pathJmbag)
    print("JMBAG:")
    for j in range(len(jmbagList)):
        ime = pathJmbag + "\\" + str(j + 1) +".jpg"
        # load the image
        img = cv2.imread(ime)[:, :, 0]
        img = np.invert(np.array([img]))
        img = img / 255

        prediction = model.predict(img)
        print("The result is probrably: " + str(np.argmax(prediction)))
        jmbag.append(np.argmax(prediction))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    # zadatak

    pathZadatki = "segmenti\\zadatak\\" + str(i + 1)

    zadatkiList = os.listdir(pathZadatki)
    print("ZADATAK:")
    for k in range(len(zadatkiList)):
        ime = pathZadatki + "\\" + str(k + 1) +".jpg"
        img = cv2.imread(ime)[:,:,0]
        img = np.invert(np.array([img]))
        img = img / 255
        prediction = model.predict(img)

        print("The result is probrably: " + str(np.argmax(prediction)))
        zadatak.append(np.argmax(prediction))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    # bodovi

    pathBodovi = "segmenti\\bodovi\\" + str(i + 1)
    print("BODOVI:")
    bodoviList = os.listdir(pathBodovi)
    for l in range(len(bodoviList)):
        ime = pathBodovi + "\\" + str(l + 1) + ".jpg"
        img = cv2.imread(ime)[:, :, 0]
        img = np.invert(np.array([img]))

        prediction = model.predict(img)
        print("The result is probrably: " + str(np.argmax(prediction)))
        bodovi.append(np.argmax(prediction))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()