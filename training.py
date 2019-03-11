#!/usr/bin/env python3

###############
# training.py #
###############

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
import numpy as np
import os

import load_data


def svm():
    '''
    SVM is trained with ALL 128-d vectors,
    and the object is serialized and saved
    '''
    model_path = "face_recognition_model/svm/model.pickle"
    label_path = "face_recognition_model/svm/label.pickle"

    # Prerocessing the data to the format of SVC
    vectors, names = load_data.vector()
    vectors = vectors.reshape(len(vectors), -1)


    le = LabelEncoder()
    labels = le.fit_transform(names)


    print("\n\n######## Start training SVM for {} 128-d vectors".format(len(names)) + " ########")

    start = time.time()

    svm = SVC(C=1.0, kernel="rbf", gamma="auto", probability=True, random_state=17)
    # svm = SVC(C=5, kernel="sigmoid", coef0=5, probability=True, random_state=17)
    svm.fit(vectors, labels)
    elapsed = time.time() - start

    print("######## SVM training completed in {0:.4f} ms ##########\n\n".format(elapsed*1000))

    with open(model_path, "wb") as f:
        pickle.dump(svm, f)

    with open(label_path, "wb") as f:
        pickle.dump(le, f)





def knn():
    '''
    KNN is trained with ALL 128-d vectors,
    and the object is serialized and saved
    '''
    model_path = "face_recognition_model/knn/model.pickle"
    label_path = "face_recognition_model/knn/label.pickle"

    # Prerocessing the data to the format of SVC
    vectors, names = load_data.vector()
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
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    knn.fit(vectors, labels)
    elapsed = time.time() - start

    print("######## KNN training completed in {0:.4f} ms ##########\n\n".format(elapsed*1000))


    with open(model_path, "wb") as f:
        pickle.dump(knn, f)


    with open(label_path, "wb") as f:
        pickle.dump(le, f)




def rf():
    '''
    Random Forest is trained with ALL 128-d vectors,
    and the object is serialized and saved
    '''
    model_path = "face_recognition_model/rf/model.pickle"
    label_path = "face_recognition_model/rf/label.pickle"

    # Prerocessing the data to the format of SVC
    vectors, names = load_data.vector()
    vectors = vectors.reshape(len(vectors), -1)

    le = LabelEncoder()
    labels = le.fit_transform(names)


    print("\n\n######## Start training Random Forest for {} 128-d vectors".format(len(names)) + " ########")

    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=17)
    rf.fit(vectors, labels)
    elapsed = time.time() - start

    print("######## Random Forest training completed in {0:.4f} ms ##########\n\n".format(elapsed*1000))


    with open(model_path, "wb") as f:
        pickle.dump(rf, f)


    with open(label_path, "wb") as f:
        pickle.dump(le, f)

