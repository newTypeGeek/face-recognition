#!/usr/bin/env python3

##################
# recognition.py #
##################

# Method to perform face recognition from 128-d vectors
# These functions are used in recognize_video.py

import numpy as np
import pickle
import time
import sys

def svm(vector, recognizer, le, max_elapsed):
    '''
    Face recognition by SVM

    Arguments:
    1. vector:      Input 128-d vector

    2. recognizer:  SVM model

    3. le:          Encoded label for SVM

    4. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Probability of SVM classification

    3. max_elapsed: (same as the 3rd argument)
 
    '''
    start = time.time()

    preds = recognizer.predict_proba(vector)[0]
    # preds = recognizer.predict(vector)[0]
    # print(preds)

    j = np.argmax(preds)
    # j = preds
    name = le.classes_[j]
    score = preds[j]
    # score = 0
    
    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed


def knn(vector, recognizer, le, max_elapsed):
    '''
    Face recognition by KNN

    Arguments:
    1. vector:      Input 128-d vector

    2. recognizer:  KNN model

    3. le:          Encoded label for KNN

    4. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Probability of KNN classification

    3. max_elapsed: (same as the 3rd argument)
 
    '''
    start = time.time()
    preds = recognizer.predict_proba(vector)[0]
    j = np.argmax(preds)
    name = le.classes_[j]
    score = preds[j]
    
    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed


def rf(vector, recognizer, le, max_elapsed):
    '''
    Face recognition by Random Forest

    Arguments:
    1. vector:      Input 128-d vector

    2. recognizer:  Random Forest model

    3. le:          Encoded label for KNN

    4. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Probability of Random Forest classification

    3. max_elapsed: (same as the 3rd argument)
 
    '''
    start = time.time()
    preds = recognizer.predict_proba(vector)[0]
    # preds = recognizer.predict(vector)[0]
    j = np.argmax(preds)
    # j = preds
    name = le.classes_[j]
    score = preds[j]
    # score = 0
    
    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed



def pearson(vector, vectors, labels, max_elapsed):
    '''
    Face recognition by searching for the
    maximum Pearson correlation with the database

    Arguments:
    1. vector:      Input 128-d vector 

    2. vectors:     128-d vectors from database
    
    3. labels:      Identities of 128-d vectors from database

    4. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Optimal value of Pearson correlation

    3. max_elapsed: (same as the 3rd argument)
    '''


    start = time.time()
    
    n = len(labels)
    idx = 0
    score = -1
    total = 0


    # This is faster than calling np.corrcoef(...) by 2 - 4 ms
    # Reason is these variables can be re-used without repeating
    # the computation in the np.corrcoef(..) function in the for loop
    vec_num = len(vector[0])
    x_mean = np.mean(vector[0])
    x_lower = np.sqrt(np.sum(vector[0]*vector[0]) - vec_num*x_mean*x_mean)

    for i in range(n):
        y_mean = np.mean(vectors[i])
        y_lower = np.sqrt(np.sum(vectors[i]*vectors[i]) - vec_num*y_mean*y_mean)

        x = ( np.dot(vector[0], vectors[i][0]) - vec_num * x_mean * y_mean ) / (x_lower * y_lower)

        # x = np.corrcoef(vector[0], vectors[i])[0][1]
        if x > score:
            score = x
            idx = i

    name = labels[idx]

    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed





def cosine(vector, vectors, labels, max_elapsed):
    '''
    Face recognition by searching for the
    maximum cosine similarity with the database

    Arguments:
    1. vector:      Input 128-d vector 

    2. vectors:     128-d vectors from database
    
    3. labels:      Identities of 128-d vectors from database

    4. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Optimal value of cosine similarity

    3. max_elapsed: (same as the 3rd argument)
    '''


    start = time.time()
    
    n = len(labels)
    idx = 0
    score = -1
    total = 0

    vector_l2 = np.sqrt(np.sum(vector[0] * vector[0]))
    
    for i in range(n):
        vectors_l2 = np.sqrt(np.sum(vectors[i] * vectors[i]))
        product_l2 = vector_l2 * vectors_l2
        x = np.dot(vector[0], vectors[i][0]) / product_l2

        if x > score:
            score = x
            idx = i

    name = labels[idx]

    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed




def l2_distance(vector, vectors, labels, max_elapsed):
    '''
    Face recognition by searching for the
    minimum L2 distance with the database

    Arguments:
    1. vector:      Input 128-d vector 

    2. vectors:     128-d vectors from database
    
    3. labels:      Identities of 128-d vectors from database

    4. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Optimal value of L2 distance

    3. max_elapsed: (same as the 3rd argument)
    '''


    start = time.time()
    
    n = len(labels)
    idx = 0
    score = sys.float_info.max
    total = 0

    
    for i in range(n):
        diff = vector[0] - vectors[i]
        x = np.sqrt( np.sum(diff * diff) )

        if x < score:
            score = x
            idx = i

    name = labels[idx]

    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed




def l1_distance(vector, vectors, labels, max_elapsed):
    '''
    Face recognition by searching for the
    minimum L1 distance with the database

    Arguments:
    1. vector:      Input 128-d vector 

    2. vectors:     128-d vectors from database
    
    3. labels:      Identities of 128-d vectors from database

 
    3. max_elapsed: Maximum time elapsed for this function
                    Used during video streaming

    Returns:
    1. name:        Identity of this vector 

    2. score:       Optimal value of L1 distance

    4. max_elapsed: (same as the 3rd argument)
    '''


    start = time.time()

    n = len(labels)
    idx = 0
    score = sys.float_info.max
    total = 0

    
    for i in range(n):
        x = np.sum( np.abs(vector[0] - vectors[i]) )
        if x < score:
            score = x
            idx = i

    name = labels[idx]

    elapsed = time.time() - start
    if elapsed > max_elapsed:
        max_elapsed = elapsed

    return name, score, max_elapsed


