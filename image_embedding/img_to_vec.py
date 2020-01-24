#!/usr/bin/env python3

#################
# img_to_vec.py #
#################

import numpy as np
import imutils
import cv2

import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from webcam import pre_recognition


def img_to_vec(image_path, detector, embedder, min_prob_filter):
    '''
    Convert an image to a 128-d vector

    Arguments:
    1. image_path:          Image path

    2. detector:            ResNet CNN

    3. embedder:            FaceNet CNN

    4. min_prob_filter:     Probability threshold to filter weak detection
                            from ResNet CNN

    Returns:
    1. vector:              128-d vector (can be None if no detection)

    '''

    # Initialize the 128-d vector
    vector = np.nan

    # Load the image and resize to width of 600 pixels,
    # while maintaing the aspect ratio. Then, get the
    # image dimension
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Generate bounding boxes for face candidates
    detections, _ = pre_recognition.locate_faces(image, detector, None)

    # Ensure at least one face candidate
    if len(detections) > 0:
        # NOTE: Assume each image has only ONE face !!!!
        #       Select the bounding box with the largest probability
        #       Here, only ONE bounding box is chosen !
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Filter weak detection
        if confidence > min_prob_filter:
            # Compute the (x, y) coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Crop the ROI and dimensions
            face = image[start_y:end_y, start_x:end_x]
            (face_h, face_w) = face.shape[:2]

            # Filter small face candidates
            if face_w >= 20 and face_h >= 20:
                # Extract the 128-d vector from the ROI
                vector, _ = pre_recognition.extract_vector(face, embedder, None)

    return vector

