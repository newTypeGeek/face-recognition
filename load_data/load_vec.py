#!/usr/bin/env python3

###############
# load_vec.py #
###############

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
