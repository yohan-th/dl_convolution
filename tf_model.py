# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    tf_model.py         #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 15:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         21/06/2019 16:33:42 #
# *************************************************************************** #

import tensorflow as tf
from tensorflow.contrib.layers import flatten


def tf_erreur(y, logits):
    # Loss - l'Erreur
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
    # reduire la moyenne de l'erreur
    loss_operation = tf.reduce_mean(cross_entropy)
    # vitesse de changement de gradiant à 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    training_operation = optimizer.minimize(loss_operation)
    return training_operation


# retourne la moyenne des vrais et faux
def tf_accuracy(y, softmax):
    # predire une preduction correcte
    predicted_cls = tf.argmax(softmax, axis=1)
    # renvoi un tableau de vrai ou faux en comparait le correcte au argmax
    correct_prediction = tf.equal(predicted_cls, tf.argmax(y, axis=1))
    # donne une moyenne
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy_operation


def create_conv(prev, filter_size, nb):
    # First convolution
    conv_w = tf.Variable(tf.truncated_normal(
        shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    conv_b = tf.Variable(tf.zeros(nb))
    # strides = nb de pixel qu'on saute pour créer le filtre
    conv = tf.nn.conv2d(prev, conv_w, strides=[1, 1, 1, 1],
                        padding='SAME') + conv_b
    # Activation: relu --> fonction d'activation choisi arbitrairement
    conv = tf.nn.relu(conv)
    # Pooling --> on reduit la taille par 2 du filtre
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')
    return conv


# Pourquoi créer un 1 couches de layer de 512 neurones après les filtres ?
# Pourquoi créer une dernière couche de 2 neurone à la fin

def create_model():
    # Placeholder
    x = tf.placeholder(tf.float32, (None, 75, 75, 3),
                       name="x")  # 75=pixel et 3=RBV
    y = tf.placeholder(tf.float32, (None, 2), name="y")
    dropout = tf.placeholder(tf.float32, None, name="dropout")

    conv = create_conv(x, 8, 32)  # 32 filtres de taille 8x8px, x=layer prec
    conv = create_conv(conv, 5, 64)
    conv = create_conv(conv, 5, 128)
    conv = create_conv(conv, 5, 256)

    # pour applatir la convolution/image dans un seul vecteur
    flat = flatten(conv)
    print(flat, flat.get_shape()[1])

    # First fully connected layer --> on rajoute un reseau de 512 neurone
    # shape=nb_neurone_couche_precedente, nb_neurone)
    fc1_w = tf.Variable(
        tf.truncated_normal(shape=(int(flat.get_shape()[1]), 512)))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1 = tf.matmul(flat, fc1_w) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # fc1 = tf.nn.dropout(fc1, keep_prob=dropout)

    # Last layer: Prediction
    fc3_w = tf.Variable(tf.truncated_normal(shape=(512, 2)))
    fc3_b = tf.Variable(tf.zeros(2))

    logits = tf.matmul(fc1, fc3_w) + fc3_b
    # permet de normaliser les valeurs de chaque neurone en 1 valeur
    softmax = tf.nn.softmax(logits)
    return x, y, dropout, softmax, logits
