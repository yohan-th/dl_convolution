# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    script.py           #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 11:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         21/06/2019 16:39:42 #
# *************************************************************************** #

from dataset import extract_dataset
from tf_model import create_model

X = []
Y = []
X, Y = extract_dataset()
model = create_model(X)
model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)

