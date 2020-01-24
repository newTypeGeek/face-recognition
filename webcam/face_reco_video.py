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
 

    # NOTE: To handle non-consecutive image naming
    # For example, the current image file names might be 
    # ['0001.jpg', '0004.jpg', '0006.jpg, 0007.jpg']
    # 
    # When taking photos, the new naming should first use this set
    # 0000.jpg, 0002.jpg, 0003.jpg, 0005.jpg
    #
    # then, followed by
    # 0008.jpg, 0009.jpg, 0010.jpg ... and so on
    
    result_path = os.path.join("result", method)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    image_exist = [x for x in os.listdir(result_path) if not x.startswith('.')]
    num_image_exist = len(image_exist)
    idx_required = []
    idx_max = 0

    # Check if there are images in result_path
    if num_image_exist > 0:
        # Convert all photos name to integer value
        idx = [0 if image == "0000.jpg" else int(os.path.splitext(image)[0].lstrip('0')) for image in image_exist]
        idx_max = max(idx)
        
        # Find the missing integer between 0 to n (inclusive)
        # These indices would be used for saving the image first
        all_idx = list(range(0, idx_max+1))
        idx_required = sorted( list(set(all_idx).difference(set(idx))) )
        idx_max = idx_max + 1


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

    while True:
        frame = vs.read()

        if frame is None:
            print("Cannot grab the video frame")
            print("Exit")
            break

        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=720)
        (h, w) = frame.shape[:2]



        score = None
        name = None

        # Locate face candidates
        detections, max_locate_time = pre_recognition.locate_faces(frame, detector, max_locate_time)

        # Loop over all detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detection
            # NOTE: This condition is NOT same
            #       as image_to_vector.py
            if confidence > min_prob_filter:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Ensure the face are large enough
                # NOTE: It furthers assume that faces
                #       should not be too small. 
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

        text = "Number of image captured = " + str(img_num)
        cv2.putText(frame_with_text, text, (50, 300), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2)
 
        text = "Press SPACEBAR to capture"
        cv2.putText(frame_with_text, text, (50, 340), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 255), 2)
 
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

        # SPACEBAR is pressed
        elif key % 256 == 32:
            if len(idx_required) > 0:
                name = str(idx_required.pop(0)).zfill(4)
            else:
                name = str(idx_max).zfill(4)
                idx_max += 1
           
            img_name = os.path.join(result_path, name + ".jpg")
            cv2.imwrite(img_name, frame)
            img_num += 1
            print("Image saved at " +  img_name)
            


    cv2.destroyAllWindows()
    vs.stop()
    
    print("Max time elapsed to locate faces     {0:.4f} ms".format(max_locate_time*1000))
    print("Max time elapsed to extract vectors  {0:.4f} ms".format(max_vector_time*1000))
    print("Max time elapsed to recognize faces  {0:.4f} ms".format(max_recog_time*1000))
