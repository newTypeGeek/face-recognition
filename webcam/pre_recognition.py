#!/usr/bin/env python3

###################
# pre_recognition #
###################


import cv2
import time


def extract_vector(face, embedder, max_elapsed=None):
    '''
    Extract 128-d vector from a face

    Arguments:
    1. face:        Image of a cropped face candidate

    2. embedder:    FaceNet CNN

    3. max_elapsed: Count time elapsed setting
                    None:   do not count time elapsed (default)

    Returns:
    1. vector:      128-d vector

    2. max_elapsed: Maximum time elapsed (can be None)
    '''
    
    if max_elapsed == None:
        faceBlob = cv2.dnn.blobFromImage(face,
                1.0/255, (96, 96), (0, 0, 0),
                swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vector = embedder.forward()
    else:
        start = time.time()

        faceBlob = cv2.dnn.blobFromImage(face,
                1.0/255, (96, 96), (0, 0, 0),
                swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vector = embedder.forward()

        elapsed = time.time() - start
        if elapsed > max_elapsed:
            max_elapsed = elapsed

    return vector, max_elapsed



def locate_faces(image, detector, max_elapsed=None):
    '''
    Locate face candidates from an image

    Arguments:
    1. image:       An image

    2. detector:    ResNet CNN


    3. max_elapsed: Count time elapsed setting
                    None:   do not count time elapsed (default)


    Returns:
    1. detections:  Bounding boxe objects
    
    2. max_elapsed: Maximum time elapsed (can be None)
    '''

    if max_elapsed == None:
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                1.0, (300, 300), (104.0, 177.0, 123.0),
                swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

    else:
        start = time.time()

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                1.0, (300, 300), (104.0, 177.0, 123.0),
                swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        elapsed = time.time() - start
        if elapsed > max_elapsed:
            max_elapsed = elapsed
    
    return detections, max_elapsed
