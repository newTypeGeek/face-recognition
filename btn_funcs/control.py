from tkinter import messagebox

import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from image_embedding.gen_vec_start import gen_vec_start
from train_clf.train_svm import train_svm
from train_clf.train_knn import train_knn
from train_clf.train_rf import train_rf


def info():
    '''
    Show the information of face recognition method when
    `Information` button is pressed.
    '''

    reference = "This face recognition demo is based on https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/\n\n"

    messagebox.showinfo("Information", reference)


def start(member):
    '''
    It is triggered when the program start:
        Clean all 128-d vectors from storage and
        Extract 128-d vectors from ALL photos
    '''
    member_num = gen_vec_start(0.3)

    # -1 is required, since one of the identities is unknown
    member.num = member_num - 1

    if member.num > 0:
        train_svm()
        train_knn()
        train_rf()


def restart(member, label):
    '''
    It is triggered when the program is running 
    and you restart it in the GUI:
        Clean all 128-d vectors from storage and
        Extract 128-d vectors from ALL photos
    '''
 
    re_init = messagebox.askquestion("Re-Initialization", "Remove all 128-d vectors and trained classifiers \n\nAre you sure?")

    if re_init == "yes":
        member_num = gen_vec_start(0.3)

        # -1 is required, since one of the identities is unknown
        member.num = member_num - 1
        
        label.config(text = "Total number of member: " + str(member.num))

        if member.num > 0:
            train_svm()
            train_knn()
            train_rf()
