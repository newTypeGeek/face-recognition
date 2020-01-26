#!/usr/bin/env python3

######################
# face_reco_video.py #
######################



from imutils.video import VideoStream
import numpy as np
import imutils
import pickle
import cv2
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from load_data import load_clf
from load_data import load_vec
from load_data import load_cnn
from webcam import pre_recognition
from webcam import recognition 


def face_reco_video(min_prob_filter, method):
    '''
    Perform face recognition from a video stream

    Arguments:
    1. min_prob_filter:     probability threshold to filter weak face candidates
                            from ResNet CNN

    2. method:              face recognition method for 128-d vectors
                            svm, knn, rf, pearson, cosine, l2, and l1

    '''


    if method == "svm":
        recognizer, le = load_clf.load_svm()
        frame_name = "Face Recognition (Support Vector Machine)"
        print("\n////////// Face Recognition (Support Vector Machine) //////////")

    elif method == "knn":
        recognizer, le = load_clf.load_knn()
        frame_name = "Face Recognition (k-Nearest Neighbours)"
        print("\n////////// Face Recognition (k-Nearest Neighbours) //////////")

    elif method == "rf":
        recognizer, le = load_clf.load_rf()
        frame_name = "Face Recognition (Random Forest)" 
        print("\n////////// Face Recognition (Random Forest) //////////")

    elif method == "pearson":
        vectors, labels = load_vec.vector()
        frame_name = "Face Recognition (Pearson Correlation)"
        print("\n////////// Face Recognition (Pearson Correlation) //////////")

    elif method == "cosine":
        vectors, labels = load_vec.vector()
        frame_name = "Face Recognition (Cosine Similarity)"
        print("\n////////// Face Recognition (Cosine Similarity) //////////")

    elif method == "l2":
        vectors, labels = load_vec.vector()
        frame_name = "Face Recognition (L2 Distance)"
        print("\n ////////// Face Recognition (L2 Distance) //////////")

    elif method == "l1":
        vectors, labels = load_vec.vector()
        frame_name = "Face Recognition (L1 Distance)"
        print("\n////////// Face Recognition (L1 Distance) //////////")

    else:
        print("Face recognition method does not exist")
        return -1

 
    # Load the ResNet and FaceNet CNN
    detector = load_cnn.resnet()
    embedder = load_cnn.facenet()

    max_locate_time = 0
    max_vector_time = 0
    max_recog_time = 0

    vs = VideoStream(src=0).start()
    cv2.namedWindow(frame_name)
    cv2.moveWindow(frame_name, 500, 200)
    img_num = 0

    # Start the webcam for face recognition
    while True:
        frame = vs.read()

        if frame is None:
            print("Cannot grab the video frame")
            print("Exit")
            break

        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=720)
        (h, w) = frame.shape[:2]

        # To be displayed on the screen
        # The score of face recognition
        # The name of the identity
        score = None
        name = None

        # Locate face candidates
        detections, max_locate_time = pre_recognition.locate_faces(frame, detector, max_locate_time)

        # Loop over all detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detection
            # NOTE: This condition is NOT same as image_embedding/img_to_vec.py
            if confidence > min_prob_filter:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Ensure the face are large enough
                # NOTE: It furthers assume that faces should not be too small. 
                if fW < 20 or fH < 20:
                    continue

                # Extract 128-d vector
                vector, max_vector_time = pre_recognition.extract_vector(face, embedder, max_vector_time)

                if method == "svm":
                    name, score, max_recog_time = recognition.svm(vector, recognizer, le, max_recog_time)
                
                elif method == "knn":
                    name, score, max_recog_time = recognition.knn(vector, recognizer, le, max_recog_time)

                elif method == "rf":
                    name, score, max_recog_time = recognition.rf(vector, recognizer, le, max_recog_time)

                elif method == "pearson":
                    name, score, max_recog_time = recognition.pearson(vector, vectors, labels, max_recog_time)
                
                elif method == "cosine":
                    name, score, max_recog_time = recognition.cosine(vector, vectors, labels, max_recog_time)

                elif method == "l2":
                    name, score, max_recog_time = recognition.l2_distance(vector, vectors, labels, max_recog_time)

                elif method == "l1":
                    name, score, max_recog_time = recognition.l1_distance(vector, vectors, labels, max_recog_time)

                else:
                    print("Face recognition method does not exist")
                    return -1

                text = "{}: {:.4f}".format(name, score)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)


        frame_with_text = np.copy(frame)

        text = "Press ESC to exit"
        cv2.putText(frame_with_text, text, (50, 380), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 128, 0), 2)
       
        cv2.imshow(frame_name, frame_with_text)


        
        # TODO: Handle unicode input
        #       Now, if the keyboard is Chinese or Russian
        #       and when the keystroke (QWER ..etc) is pressed
        #       the program stops with assertion error
        # key = cv2.waitKey(1)
        key = cv2.waitKeyEx(1)

        # ESC is pressed
        if key % 256 == 27:
            break


    cv2.destroyAllWindows()
    vs.stop()
    
    print("Max time elapsed to locate faces     {0:.4f} ms".format(max_locate_time*1000))
    print("Max time elapsed to extract vectors  {0:.4f} ms".format(max_vector_time*1000))
    print("Max time elapsed to recognize faces  {0:.4f} ms".format(max_recog_time*1000))
