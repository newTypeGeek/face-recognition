# webcam
`webcam` package consists of functions to perform livestream face recognition from **webcam**


1. `face_reco_video.py`<br>
   1. `face_reco_video` selects a classifier chosen from GUI, and then start a video stream to perform face detection and face recognition.

2. `pre_recognition.py`<br>
   1. `extract_vector`: extract 128-d vector from a **face**
   2. `loacate_faces`: locate the face from a photo and crop it out

3. `recognition.py`<br>
   Perform face recognition using different classifiers<br>
   1. `svm`
   2. `knn`
   3. `rf`
   4. `pearson`
   5. `cosine`
   6. `l2_distance`
   7. `l1_distance`

4. `take_photos.py`<br>
   1. `take_photos`: take photos from the webcam and save them
   
