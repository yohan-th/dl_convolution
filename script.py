# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    script.py           #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 11:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         2019/10/04 16:34:52 #
# *************************************************************************** #

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.utils import shuffle
import sys

path_1 = "./datasets/cat&dog/cat/"
path_2 = "./datasets/cat&dog/dog/"

X = []
Y = []
def create_data(path):
    category = 1 if re.match(".*/cat/.*", path) else 0
    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        Y.append(category)

create_data(path_2)
create_data(path_1)

X, Y = shuffle(X, Y)
X = np.array(X).reshape(-1, 80,80,1)
Y = np.array(Y)
X = X/255

model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, epochs=20, batch_size=32, validation_split=0.2)

global_result = []
n = 0
train_dir = "./datasets/cat&dog/train/"
for p in os.listdir(train_dir):
    n += 1
    reality = p.split(".")[0]

    img_array = cv2.imread(os.path.join(train_dir,p),cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(80, 80))
    #plt.imshow(new_img_array, cmap="gray")
    #plt.show()
    new_img_np = np.array(new_img_array).reshape(-1,80,80,1)
    new_img_np = new_img_np/255
    prediction = model.predict_on_batch(new_img_np)
    if round(prediction[0][0]) == 1:
        pred_target = "cat"
    else:
        pred_target = "dog"
    result = " True " if reality == pred_target else " False "
    if reality == pred_target:
        global_result.append(1)
    else:
        global_result.append(0)

    accuracy = sum(global_result) / n
    #print("Prediction à "+str(prediction[0][0])+str(result)+"Accuracy global:"+str(accuracy))
print(accuracy)

