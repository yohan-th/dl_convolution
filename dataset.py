# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    dataset.py          #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 09:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         2019/07/31 14:52:18 #
# *************************************************************************** #

import glob
import cv2
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def extract_dataset():
    training_data = []

    files = glob.glob('datasets/flowers/rose/*.jpg')

    for file in files:
        img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        label = 1
        training_data.append([new_img_array,label])

    files = glob.glob('datasets/car&truck/car/*.jpg')

    for file in files:
        img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        label = 0
        training_data.append([new_img_array, label])

    shuffle(training_data)
    shuffle(training_data)

    X = []
    Y = []

    for elem in training_data:
        X.append(elem[0])
        Y.append(elem[1])

    X = np.array(X).reshape(-1, 80, 80, 1)
    Y = np.array(Y)
    #plt.imshow(training_data[0][0], cmap="gray")
    #plt.show()

    return X, Y
