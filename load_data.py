#!/usr/bin/env python3

################
# load_data.py #
################


import pickle
import os
import numpy as np



def vector():
    '''
    Load the 128-d vectors which are extracted from dataset images
    '''
    path = "embeddings/data/overall/embeddings.pickle"
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Get only the names from the key
    names = [key.split(os.sep)[-2] for key in data.keys()]

    # Get the numerical vector from value
    vectors = np.array(list(data.values()))
    
    # NOTE:
    # names have repeating entries. Therefore, you cannot create 
    # dicts from names and vectors

    return vectors, names



def svm():
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



def knn():
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


def rf():
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

