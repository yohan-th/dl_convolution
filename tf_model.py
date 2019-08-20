# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    tf_model.py         #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   2019/07/31 14:52:40 #
#                                                                             #
#         Contact: yohan.thollet@gfi.world                Updated by yohan    #
#                                                         2019/07/31 14:52:40 #
# *************************************************************************** #

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

def create_model(X):
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
    return model