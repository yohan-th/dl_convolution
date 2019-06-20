import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import extract_dataset, show_images
from tf_model import create_model, tf_erreur, tf_accuracy
from tf_training import train_model


features, targets = extract_dataset()
#show_images(features, 5)


# x=feature y=label
x_train, x_valid, y_train, y_valid = train_test_split(features, targets,
                                                      test_size=0.05,
                                                      random_state=42)

print("X_train.shape", x_train.shape)
print("X_valid.shape", x_valid.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_valid.shape)

x, y, dropout, softmax, logits = create_model()

training_operation = tf_erreur(y, logits)
accuracy_operation = tf_accuracy(y, softmax)

batch_size = 255

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_model(batch_size, sess, x, y, x_train, y_train, x_valid, y_valid,
            dropout, accuracy_operation, training_operation)
