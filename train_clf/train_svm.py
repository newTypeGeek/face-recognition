#!/usr/bin/env python3

################
# train_svm.py #
################

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import time
import numpy as np

import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from load_data import load_vec


def train_svm():
    '''
    SVM is trained with ALL 128-d vectors,
    and the object is serialized and saved
    '''
    model_path = "face_recognition_model/svm/model.pickle"
    label_path = "face_recognition_model/svm/label.pickle"

    # Prerocessing the data to the format of SVC
    vectors, names = load_vec.vector()
    vectors = vectors.reshape(len(vectors), -1)


    le = LabelEncoder()
    labels = le.fit_transform(names)


    print("\n\n######## Start training SVM for {} 128-d vectors".format(len(names)) + " ########")

    start = time.time()

    clf = SVC(C=1.0, kernel="rbf", gamma="auto", probability=True, random_state=17)
    clf.fit(vectors, labels)
    elapsed = time.time() - start

    print("######## SVM training completed in {0:.4f} ms ##########\n\n".format(elapsed*1000))

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    with open(label_path, "wb") as f:
        pickle.dump(le, f)
