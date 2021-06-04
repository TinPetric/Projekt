import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    # reshape

    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.fit(x_train, y_train, epochs=3)

    val_loss,val_acc = model.evaluate(x_test,y_test)
    print("loss-> ",val_loss,"\nacc-> ",val_acc)

    predictions = model.predict([x_test])
    print('lable -> ', y_test[2])
    print('prediction -> ', np.argmax(predictions[2]))

    #draw(x_test[2])

    # saving the model
    # .h5 or .model can be used

    model.save('epic_num_reader.h5')

def clasify(img):
    new_model = tf.keras.models.load_model('epic_num_reader.h5')
    #path="segmenti/0bodovi1.jpg"
    #img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)

    img=cv2.resize(img,(28,28))
    cv2.imshow("1",img)
    #draw(img)
    img=np.array(img)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    img=img.flatten()

    img = tf.keras.utils.normalize(img, axis=0)
    predictions = new_model.predict(img)

    print(predictions)
    print('prediction -> ', np.argmax(predictions))
    return np.argmax(predictions)

