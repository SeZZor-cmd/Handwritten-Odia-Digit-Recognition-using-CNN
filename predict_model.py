# For AVX and SSE instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation

import matplotlib.font_manager as fm



# Image attributes
img_width, img_height = 28, 28
num_classes = 10
CHANNELS = 1

# Function to create model
def create_model():
    model = Sequential()

    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(img_height, img_width, CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), padding="same", input_shape=(img_height, img_width, CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dense(num_classes, activation='softmax'))


    return model

test_folder = 'test_images'

c = 1
for i in os.listdir(test_folder):
    # create a grid of 3x3 images
    plt.subplot(5, 5, c)
    c += 1
    image = cv2.imread(test_folder + "\\" + i, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = np.array(image).reshape((img_width, img_height, 1))
    plt.imshow(image.reshape(28, 28), cmap=plt.get_cmap('gray'))

    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)

    # Creating and loading model
    model = create_model()
    # Load trained nueral nets
    model.load_weights('./out/mnist.h5')

    prediction = model.predict(image)[0]


    import pandas as pd

    def get_char(classint):
        df = pd.read_csv('Numbers.csv')

        for index, row in df.iterrows():
            if row['class'] == classint:
                return row['char']
            else:
                pass

    def percent(num1, num2):
        num1 = float(num1)
        num2 = float(num2)
        percentage = '{0:.2f}'.format((num1 / num2 * 100))
        return percentage

    pred_dict = {}
    i = 0
    for listitem in list(prediction):
        pred_dict[i] = listitem
        i += 1

    for w in sorted(pred_dict, key=pred_dict.get, reverse=True)[:1]:
        prediction_label = get_char(w)
        prediction_conf =  percent(pred_dict[w], 1)

    plt.tight_layout()

    plt.xlabel(prediction_label)
    plt.ylabel(prediction_conf)
# # show the plot
plt.show()

