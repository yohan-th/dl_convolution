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

def train_model(batch_size, sess, x, y, x_train, y_train, x_valid, y_valid,
                dropout, accuracy_operation, training_operation):
    i = 0
    for epoch in range(0, 10000):
        print(">> Epoch: %s" % epoch)
        # Shuffle --> on mélange les images sans perdre la relation en x et y
        indexs = np.arange(len(x_train))
        np.random.shuffle(indexs)
        X_train = x_train[indexs]
        y_train = y_train[indexs]

        for b in range(0, len(x_train), batch_size):
            batch = augmented_batch (x_train[b:b + batch_size])
            # batch = X_train[b:b+batch_size]

            if i % 20 == 0:
                # print(sess.run(predicted_cls, feed_dict={dropout: 1.0, x: batch, y: y_train[b:b+batch_size]}))
                print("Accuracy [Train]:", sess.run(accuracy_operation,
                                                    feed_dict={dropout: 1.0,
                                                    x: batch, y: y_train[b:b + batch_size]}))
            sess.run(training_operation, feed_dict={dropout: 0.8, x: batch,
                                                    y: y_train[b:b + batch_size]})
            i += 1
        if epoch % 2 == 0:
            accs = []
            for b in range(0, len(x_valid), batch_size):
                accs.append(sess.run(accuracy_operation, feed_dict={dropout: 1.,
                                                                    x: x_valid[
                                                                       b:b + batch_size],
                                                                    y: y_valid[
                                                                       b:b + batch_size]}))
            print("Accuracy [Validation]", np.mean(accs))