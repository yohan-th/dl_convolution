# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    script.py           #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 11:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         21/06/2019 16:39:42 #
# *************************************************************************** #

import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import extract_dataset, show_images
from tf_model import create_model, tf_erreur, tf_accuracy
from tf_training import train_model
from all_class import c_dataset, c_model, c_train


features, targets = extract_dataset()
# show_images(features, 5)

# train_test_split permet de split la dataset en 2 pour avoir une partie train
# et un autre pour valider
# 10% de la dataset est dédié a la validation et 90% à l'entrainement
x_train, x_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.1)
dtst = c_dataset(x_train, x_valid, y_train, y_valid)

print("X_train.shape", dtst.x_train.shape)
print("X_valid.shape", dtst.x_valid.shape)
print("y_train.shape", dtst.y_train.shape)
print("y_valid.shape", dtst.y_valid.shape)

x, y, dropout, softmax, logits = create_model()
modl = c_model(x, y, dropout, softmax, logits)

training_operation = tf_erreur(modl.y, modl.logits)
accuracy_operation = tf_accuracy(modl.y, modl.softmax)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# lot de 100 image qu'on envoi dans le graphe tenserflow
train = c_train(100, sess, accuracy_operation, training_operation)

train_model(dtst, modl, train)
