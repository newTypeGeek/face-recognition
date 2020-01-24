#!/usr/bin/env python3

###############
# train_rf.py #
###############

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
import numpy as np

import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from load_data import load_vec


def train_rf():
    '''
    Random Forest is trained with ALL 128-d vectors,
    and the object is serialized and saved
    '''
    model_path = "face_recognition_model/rf/model.pickle"
    label_path = "face_recognition_model/rf/label.pickle"

    # Prerocessing the data to the format of SVC
    vectors, names = load_vec.vector()
    vectors = vectors.reshape(len(vectors), -1)

    le = LabelEncoder()
    labels = le.fit_transform(names)


    print("\n\n######## Start training Random Forest for {} 128-d vectors".format(len(names)) + " ########")

    start = time.time()
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=17)
    clf.fit(vectors, labels)
    elapsed = time.time() - start

    print("######## Random Forest training completed in {0:.4f} ms ##########\n\n".format(elapsed*1000))


    with open(model_path, "wb") as f:
        pickle.dump(clf, f)


    with open(label_path, "wb") as f:
        pickle.dump(le, f)

