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

## AN UPDATED README IS IN PROGRESS

# Project Structure
1. `main.py`<br>
   The main program to be executed by Python Interpreter. It consists of `Tkinter` GUI design.

2. `btn_funcs`<br>
   A package consists of button functions of GUI

3. `image_embedding`<br>
    A package consists of **core** functions:
    1. Starting (re-starting) the `main.py` program
    2. Register new members
    3. Add new photos

4. `webcam`<br>
   A package using webcam to perform *photo taking* and *face recognition*

5. `train_clf`<br>
   A package to train machine learning models (classifiers)

6. `load_data`<br>
   A package to load 128-d vectors, trained machine learning models, and pre-trained CNN models
