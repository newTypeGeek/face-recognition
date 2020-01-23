import image_to_vector
import training
from tkinter import messagebox


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
    member_num = image_to_vector.gen_vector_initialize(0.3)

    # -1 is required, since one of the identities is unknown
    member.num = member_num - 1

    if member.num > 0:
        training.svm()
        training.knn()
        training.rf()


def restart(member, label):
    '''
    It is triggered when the program is running 
    and you restart it in the GUI:
        Clean all 128-d vectors from storage and
        Extract 128-d vectors from ALL photos
    '''
 
    re_init = messagebox.askquestion("Re-Initialization", "Remove all machine learning files in `output` directory\nRe-scan all photos in `dataset` directory\n\nAre you sure?")

    if re_init == "yes":
        member_num = image_to_vector.gen_vector_initialize(0.3)

        # -1 is required, since one of the identities is unknown
        member.num = member_num - 1
        
        label.config(text = "Total number of member: " + str(member.num))

        if member.num > 0:
            training.svm()
            training.knn()
            training.rf()
