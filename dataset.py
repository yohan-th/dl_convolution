import glob
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt


def show_images(features, nb):
    for a in [randint(0, len(features)) for _ in range(nb)]:
        plt.imshow(features[a], cmap="gray")
        plt.show()


def extract_dataset():
    targets = []
    features = []

    files_cat = glob.glob('train/cat/*.jpg')

    for file in files_cat:
        features.append(np.array(Image.open(file).resize((75, 75))))
        target = [1, 0]
        targets.append(target)

    files_cat = glob.glob('train/dog/*.jpg')

    for file in files_cat:
        features.append(np.array(Image.open(file).resize((75, 75))))
        target = [0, 1]
        targets.append(target)

    features = np.array(features)
    targets = np.array(targets)

    print("features shape", features.shape)
    print("Targets shape", targets.shape)
    return features, targets
