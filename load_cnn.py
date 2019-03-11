#!/usr/bin/env python3

import cv2


def resnet():
    '''
    Load the face detection model

    Returns:
    1. detector:    ResNet CNN
    '''
    proto_path = "face_detection_model/deploy.prototxt"
    model_path = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

    print("Loading the CNN (ResNet+SSD) for face detection")
    detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    return detector


def facenet():
    '''
    Load the embedding model

    Returns:
    1. embedder:    FaceNet CNN
    '''
    embedding_model = "embeddings/model/openface_nn4.small2.v1.t7"

    print("Loading the CNN (FaceNet) for embedding")
    embedder = cv2.dnn.readNetFromTorch(embedding_model)

    return embedder

