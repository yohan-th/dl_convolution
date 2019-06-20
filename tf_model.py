import tensorflow as tf
from tensorflow.contrib.layers import flatten


def tf_erreur(y, logits):
    # Loss - l'Erreur
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
    loss_operation = tf.reduce_mean(
        cross_entropy)  # reduire la moyenne de l'erreur
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    training_operation = optimizer.minimize(loss_operation)
    return (training_operation)

def tf_accuracy(y, softmax):
    predicted_cls = tf.argmax(softmax, axis=1)
    correct_prediction = tf.equal(predicted_cls, tf.argmax(y, axis=1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (accuracy_operation)

def create_conv(prev, filter_size, nb):
    # First convolution
    conv_w = tf.Variable(tf.truncated_normal(
        shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    conv_b = tf.Variable(tf.zeros(nb))
    # strides = nb de pixel qu'on saute pour créer le fltre
    conv = tf.nn.conv2d(prev, conv_w, strides=[1, 1, 1, 1],
                        padding='SAME') + conv_b
    # Activation: relu --> choix arbitraire de cette fonction d'activation
    conv = tf.nn.relu(conv)
    # Pooling --> on reduit la taille par 2 du filtre
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')
    return conv


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
    conv = create_conv(conv, 5, 215)

    flat = flatten(conv)
    # pour applatir la convolution/image dans un seul vecteur
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

    softmax = tf.nn.softmax(logits)
    return x, y, dropout, softmax, logits
