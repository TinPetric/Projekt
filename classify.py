import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import openpyxl




# Give the location of the file
loc = ("oznake.xlsx")

wb = openpyxl.load_workbook(loc)
sheet = wb.active

studenti=[]

max_col = sheet.max_column
max_row = sheet.max_row

for i in range(1, max_row + 1):
    student=[]
    for j in range(1, max_col + 1):
        cell = sheet.cell(row=1, column=j)
        cell_obj = sheet.cell(row = i, column = j)
        student.append({cell.value : cell_obj.value})
    studenti.append((student))




def train():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)


    def draw(n):
        plt.imshow(n, cmap=plt.cm.binary)
        plt.show()


    #draw(x_train[0])

    # there are two types of models
    # sequential is most common, why?

#    model = tf.keras.models.Sequential()
#
#    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#    # reshape
#
#    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

#    sample_shape = x_train[0].shape
#    print(sample_shape)
#    img_width, img_height = sample_shape[0], sample_shape[1]
#    input_shape = (img_width, img_height, 1)
#
#    x_train = x_train.reshape(len(x_train), input_shape[0], input_shape[1], input_shape[2])
#    x_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])
#    #input_shape = (len(x_train),img_width, img_height, 1)
#    model = tf.keras.Sequential([
#        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape),
#        tf.keras.layers.MaxPooling2D((2, 2)),
#        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
#        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.fit(x_train, y_train, epochs=2)

   # val_loss,val_acc = model.evaluate(x_test,y_test)
   # print("loss-> ",val_loss,"\nacc-> ",val_acc)

   # predictions = model.predict([x_test])
   # print('lable -> ', y_test[2])
   # print('prediction -> ', np.argmax(predictions[2]))

    #draw(x_test[2])

    # saving the model
    # .h5 or .model can be used

    model.save('epic_num_reader.h5')

def clasify():

    new_model = tf.keras.models.load_model('epic_num_reader.h5')
    #path="segmenti/0bodovi1.jpg"
    #img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)

   # img=cv2.resize(img,(28,28))
    #draw(img)

    list = os.listdir("segmenti2")
    path = 'segmenti2\\'


    test_images=[]
    for i in list:
        if "jmbag" in i:
            print(i)

            img = cv2.imread(path + str(i))[:, :, 0]
            cv2.imshow("",img)     
            img = cv2.resize(img, (28, 28))
            img = np.invert(np.array([img]))
            img = img / 255
            #test_images.append((img))
            predictions=new_model.predict(img)
            print(np.argmax(predictions))
   # for i in predictions:
    #    print(np.argmax(i))
#
# im2 = cv2.imread("test.jpg")[:, :, 0]
# cv2.imshow("1", im2)
# im2 = cv2.resize(im2, (28, 28))
# im2 = np.invert(np.array([im2]))
# im2 = im2 / 255
# #im2 = cv2.resize(im2, (28, 28))
# predictions = new_model.predict(im2)
# print(np.argmax(predictions))
#
# im2 = cv2.imread("test2.jpg")[:, :, 0]
#
# im2 = cv2.resize(im2, (28, 28))
# im2 = np.invert(np.array([im2]))
# im2 = im2 / 255
# # im2 = cv2.resize(im2, (28, 28))
# predictions = new_model.predict(im2)
#


    #print(np.argmax(predictions))
train()
clasify()