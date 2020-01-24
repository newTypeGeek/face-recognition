#!/usr/bin/env python3

################
# train_knn.py #
################

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time
import numpy as np

import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from load_data import load_vec


def train_knn():
    '''
    KNN is trained with ALL 128-d vectors,
    and the object is serialized and saved
    '''
    model_path = "face_recognition_model/knn/model.pickle"
    label_path = "face_recognition_model/knn/label.pickle"

    # Prerocessing the data to the format of SVC
    vectors, names = load_vec.vector()
    vectors = vectors.reshape(len(vectors), -1)

    le = LabelEncoder()
    labels = le.fit_transform(names)


    print("\n\n######## Start training KNN for {} 128-d vectors".format(len(names)) + " ########")

    # n_neighors cannot be greater than number of data point
    label_num = len(labels)
    if label_num < 5:
        n = label_num
    else:
        n = 5

    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=n, weights='distance')
    clf.fit(vectors, labels)
    elapsed = time.time() - start

    print("######## KNN training completed in {0:.4f} ms ##########\n\n".format(elapsed*1000))


    with open(model_path, "wb") as f:
        pickle.dump(clf, f)


    with open(label_path, "wb") as f:
        pickle.dump(le, f)
