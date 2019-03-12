# face-recognition
Perform face recognition in video stream for registered members. This repo is based on https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/<br>

The face recognition pipeline can be summarized as follow. <br>
1. When an video frame is fed into the system, it first performs **face detection** using a pre-trained Convolution Neural Network (CNN) model (ResNet + SSD) and returns a set of Region Of Interest (ROI). <br>

2. Then the cropped face images are passed to another pre-trained CNN model to perform **face encoding** (FaceNet architecture trained by OpenFace) and returns a list of 128-dimension vectors for each face.<br>

3. Finally, **face recognition** is performed on the 128-dimension vectors using various methods. In this repo, we use machine learning methods including **Support Vector Machine (SVM)**, **k-Nearest Neighbours (k-NN)**, and **Random Forest**. Moreover, we also use non-machine learning methods -- **Pearson correlation**, **Cosine similarity**, **L2 distance**, and **L1 distance**.

# Installation
## Pre-requisite
The python scripts are written for Python 3 only and requires the following modules <br>
1. numpy
2. sklearn
3. opencv-python
4. imutils
5. tkinter

Install the first 4 modules using `pip`
```
$ pip install numpy
$ pip install sklearn
$ pip install opencv-python
$ pip install imutils
```

Installation of tkinter is a bit tricky since it is not included in `pip`. But if you are using Anaconda Python distribution, you should have it installed already. Otherwise check out these posts.

### Mac
<https://stackoverflow.com/questions/36760839/why-my-python-installed-via-home-brew-not-include-tkinter>

### Linux
<https://stackoverflow.com/questions/4783810/install-tkinter-for-python>


## Clone this repo
`$ git clone https://github.com/newTypeGeek/face-recognition`

# How to run
`$ python3 main.py`

Then you will see a GUI as shown below with *Total number of member = 0*.<br>

<img src="https://github.com/newTypeGeek/face-recognition/blob/master/gui.png" width="360">

1. **Registration** button registers a member given the valid *First Name* and *Last Name* input. Next, user is asked to
   take up to 10 photos to complete the registration. The photos would be stored in `dataset/<your_name>`. Then, the `Total number of member` in GUI should be incremented by one.

2. **Add photos** button takes additional for an existing member given the First Name and Last Name input.

3. **Face Recognition (method)** button starts performing face recognition from a video stream using the *method*.

4. **Information** button shows the reference of this program.

5. **Restart** button clears and re-generates all the `embeddings.pickle` and `face_recognition_model` files, based on the current images in the `dataset` directory.
*This button is useful if user adds or delete photos without interacting with the GUI.
This procedure also occurs when the program starts (i.e. `$ python3 main.py`)*

6. **Quit** button closes the program immediately.

## Understanding the scripts
1. `image_to_vector.py`<br>
It is called when the program just starts (`$ python3 main.py`) or **Restart** button is pressed or **Registration** button is pressed and completed or  **Add photos** button is pressed and completed. This script consists of function that convert images in `dataset` to 128-d vectors and stored in the corresponding directories under `embeddings/data`.

2. `load_cnn.py` <br>
It consists of functions to load the pre-trained CNN models for face detection and face encoding.

3. `load_data.py` <br>
It consists of functions to load all the 128-d vectors, and other trained machine learning models (SVM, k-NN, Random Forest).

4. `main.py`<br>
It is the main script, setting up the GUI as shown above.

5. `pre_recognition.py`<br>
It consists of functions for face detection and face encoding.

6. `recognition.py` <br>
It consists of various methods for actual face recognition using 128-d vectors. Methods include SVM, k-NN, Random Forest, Pearson correlation, Cosine similarity, L2 distance, and L1 distance. This script is called when **Face Recognition (method)** button is pressed.

7. `recognize_video.py` <br>
It turns on the camera and start face recognition from video stream when **Face Recognition (method)** button is pressed. Detected faces would be surrounded by red bounding boxes, with the identity and score displayed.

8. `take_photos.py` <br>
It turns on the camera for taking photos *manually* when a new member is registered or adding photos to an existing member. The photos would be stored in `dataset/<your_name>` directory. This script is called when **Registration** button is pressed and completed or **Add photos** is pressed and completed.

9. `training.py` <br>
It consists of three functions to train different machine learning models (i.e. SVM, k-NN, and Random Forest). These models are trained (or re-trained) only if *Total number of member > 0*. The trained model would be saved in `face_recognition_model/<method>` directory. These trained model would be used for face recognition from a video stream.
This script is called when the program just starts (`$ python3 main.py`) or **Restart** button is pressed or **Registration** button is pressed and completed or **Add photos** button is pressed and completed. 


## Understanding the directories
1. `dataset` <br> 
It consists of all the images from all registered members + `unknown`.
All images are named from `0000` to `9999`.
At the beginning, you should only see the `unknown` directory.
In this program `unknown` identity is required and it is not regarded as a member.
The images are generated by some GANs.
    ```
    dataset
    └── unknown
        ├── 0000.png
        ├── 0001.png
        ├── 0002.png
        ├── 0003.png
        └── 0004.png
    ```


2. `embeddings` <br>
    It consists of the embedding model and data of 128-d vectors
    - `embeddings.pickle` is the serialized Python dictionary with keys being the image paths, and values being 128-d vectors
    - `embeddings/data/individual/<identity>/embeddings.pickle` stores the 128-d vectors
      corresponds to that <identity>
    - `embeddings/data/overall/embeddings.pickle` stores the 128-d vectors of ALL <identity>
    - `openface_nn4.small2.v1.t7` is the pre-trained FaceNet CNN to extract 128-d vectors from a face
    ```
    embeddings
    ├── data
    │   ├── individual
    │   │   └── unknown
    │   │       └── embeddings.pickle
    │   └── overall
    │       └── embeddings.pickle
    └── model
        └── openface_nn4.small2.v1.t7
    ```
    <br><br>

3. `face_detection_model`<br>
    It consists of the pre-trained model which is used to locate the face candidates from an image
    - `deploy.prototxt` is the architecture of the CNN.
    - `res10_300x300_ssd_iter_140000.caffemodel` is the pre-trained weights of this CNN.
    ```
    face_detection_model
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
    ```
    <br><br>

4. `face_recognition_model`<br>
    It *will* consist of trained face recognition model
    - `label.pickle` is the serialized encoding of the identity
    - `model.pickle` is the serialized model<br>
    ```
    face_recognition_model
    ├── knn
    ├── rf
    └── svm
    ```
Rightnow, the sub-directories `knn` (k-Nearest Neighbour), `rf` (Random Forest), and `svm` (Support Vector Machine) are empty.
The `label.pickle` and `model.pickle` files will be generated when the *Total number of member > 0*. <br><br>

5. `result`<br>
    It consists of manually captured images when performing face recognition from video stream 
    using various face recognition method. Rightnow, the sub-directories are empty.
    ```
    result
    ├── cosine
    ├── knn
    ├── l1
    ├── l2
    ├── pearson
    ├── rf
    └── svm
    ```
