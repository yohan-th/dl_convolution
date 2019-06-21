# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    all_class.py        #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   21/06/2019 13:64:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         21/06/2019 16:33:42 #
# *************************************************************************** #

class c_dataset(object):
    def __init__(self, x_train, x_valid, y_train, y_valid):
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid


class c_model(object):
    def __init__(self, x, y, dropout, softmax, logits):
        self.x = x
        self.y = y
        self.dropout = dropout
        self.softmax = softmax
        self.logits = logits


class c_train(object):
    def __init__(self, batch_size, sess, accuracy_operation, training_operation):
        self.btch_size = batch_size
        self.sess = sess
        self.acc_ope = accuracy_operation
        self.train_ope = training_operation
