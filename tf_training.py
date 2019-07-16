# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    tf_training.py      #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 11:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         21/06/2019 15:39:42 #
# *************************************************************************** #

from random import randint
import random
import numpy as np
from PIL import Image


def augmented_batch(batch):
    """
    """
    n_batch = []

    for img in batch:
        if random.uniform(0, 1) > 0.75:
            process_img = Image.fromarray(
                np.uint8(img.reshape(75, 75, 3))).rotate(randint(-45, 45))
            n_img = np.array(process_img)
            n_batch.append(n_img.reshape(75, 75, 3))
        else:
            n_batch.append(img)

    return n_batch


# On mélange les images sans perdre les targets (chat ou chien)
# Pour éviter de retomber sur les meme images
def shuffle_images(dataset):
    indexs = np.arange(len(dataset.x_train))
    np.random.shuffle(indexs)
    dataset.x_train = dataset.x_train[indexs]
    dataset.y_train = dataset.y_train[indexs]
    return dataset


def run_trainning(i, dtst, modl, t):
    for b in range(0, len(dtst.x_train), t.btch_size):
        batch = augmented_batch(dtst.x_train[b:b + t.btch_size])
        # batch = X_train[b:b+batch_size]

        if i % 20 == 0:
            acc_train = t.sess.run(t.acc_ope,
                                     feed_dict={
                                         modl.dropout: 1.0,
                                         modl.x: batch,
                                         modl.y: dtst.y_train[b:b + t.btch_size]})
            print("Accuracy [Train]:", acc_train)
        t.sess.run(t.train_ope,
                      feed_dict={
                          modl.dropout: 0.8, #acceleration apprentissage
                          modl.x: batch,
                          modl.y: dtst.y_train[b:b + t.btch_size]})
        i += 1


def check_accuracy(dtst, modl, t):
    accs = []
    for b in range(0, len(dtst.x_valid), t.btch_size):
        accs.append(t.sess.run(t.acc_ope,
                                  feed_dict={
                                      modl.dropout: 1.0,
                                      modl.x: dtst.x_valid[b:b + t.btch_size],
                                      modl.y: dtst.y_valid[b:b + t.btch_size]}))
    print("Accuracy [Validation]", np.mean(accs))


# dataset = [x_train, x_valid, y_train, y_valid]
# model = [x, y, dropout, softmax, logits]
# t = [batch_size, sess, accuracy_operation, training_operation]
def train_model(dataset, model, t):
    i = 0
    for epoch in range(0, 100):
        print(">> Epoch: %s" % epoch)
        dataset = shuffle_images(dataset)

        run_trainning(i, dataset, model, t)
        if epoch % 2 == 0:
            check_accuracy(dataset, model, t)
