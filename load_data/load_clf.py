#!/usr/bin/env python3

###############
# load_clf.py #
###############


import pickle
import os
import numpy as np


def load_svm():
    '''
    Load the trained SVM model
    '''
    model_path = "face_recognition_model/svm/model.pickle"
    label_path = "face_recognition_model/svm/label.pickle"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(label_path, "rb") as f:
        label = pickle.load(f)
    
    return model, label



def load_knn():
    '''
    Load the trained k-NN model
    '''
    model_path = "face_recognition_model/knn/model.pickle"
    label_path = "face_recognition_model/knn/label.pickle"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(label_path, "rb") as f:
        label = pickle.load(f)
    
    return model, label


def load_rf():
    '''
    Load the trained Random Forest model
    '''
    model_path = "face_recognition_model/rf/model.pickle"
    label_path = "face_recognition_model/rf/label.pickle"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(label_path, "rb") as f:
        label = pickle.load(f)
    
    return model, label

