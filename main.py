#!/usr/bin/env python3

##############
#  main.py  #
##############

# Uncomment these lines if you are planning to 
# build a standalone executable using PyInstaller
# import multiprocessing
# multiprocessing.freeze_support()



import tkinter as tk
from tkinter import messagebox
import string
import os

import training
import image_to_vector
import take_photos
import recognize_video



def information():
    '''
    Show the information of face recognition method when
    `Information` button is pressed.
    '''

    reference = "This face recognition demo is based on https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/\n\n"

    messagebox.showinfo("Information", reference)




def face_recognition_l1():
    '''
    Perform face recognition using Pearson correlation when `Face Recognition (L1 distance) is pressed
    '''
    recognize_video.face_recognition(0.8, "l1")


def face_recognition_l2():
    '''
    Perform face recognition using Pearson correlation when `Face Recognition (L2 distance) is pressed
    '''
    recognize_video.face_recognition(0.8, "l2")



def face_recognition_cosine():
    '''
    Perform face recognition using Pearson correlation when `Demo (Cosine)` is pressed
    '''
    recognize_video.face_recognition(0.8, "cosine")



def face_recognition_pearson():
    '''
    Perform face recognition using Pearson correlation when `Face Recognition (Pearson) is pressed
    '''
    recognize_video.face_recognition(0.8, "pearson")



def face_recognition_svm():
    '''
    Perform face recognition using SVM when `Face Recognition (SVM) is pressed
    '''
    global member_num

    if member_num > 0:
        recognize_video.face_recognition(0.8, "svm")
    else:
        messagebox.showerror("Error", "SVM requires at least one registered member")


def face_recognition_knn():
    '''
    Perform face recognition using KNN when `Face Recognition (KNN) is pressed
    '''
    global member_num

    if member_num > 0:
        recognize_video.face_recognition(0.8, "knn")
    else:
        messagebox.showerror("Error", "KNN requires at least one registered member")



def face_recognition_rf():
    '''
    Perform face recognition using Random Forest when `Face Recognition (Random Forest) is pressed
    '''
    global member_num

    if member_num > 0:
        recognize_video.face_recognition(0.8, "rf")
    else:
        messagebox.showerror("Error", "Random Forest requires at least one registered member")




def initialization():
    '''
    Initialization by extracting 128-d vectors from ALL photos
    Count the total number of members
    '''


    global member_num
    member_num = image_to_vector.gen_vector_initialize(0.3)

    # -1 is required, since one of the identities is unknown
    member_num = member_num - 1


    if member_num > 0:
        training.svm()
        training.knn()
        training.rf()



def re_initialization():
    '''
    Initialization by extracting 128-d vectors from ALL photos
    Count the total number of members
    '''

    re_init = messagebox.askquestion("Re-Initialization", "Remove all machine learning files in `output` directory\nRe-scan all photos in `dataset` directory\n\nAre you sure?")
    
    global member_num

    if re_init == "yes":

        member_num = image_to_vector.gen_vector_initialize(0.3)

        # -1 is required, since one of the `unknown` directory is not 
        # regarded as an member
        member_num = member_num - 1

        label.config(text = "Total number of member: " + str(member_num))

        if member_num > 0:
            training.svm()
            training.knn()
            training.rf()


def check_name(name, which):
    '''
    Check if the input name is valid or not.
    1. No special character is allowed
    2. All whitespace at the beginning and the end of 
       the name is removed
    '''

    invalid_char = set(string.punctuation)

    # Filter out special characters
    if any(char in invalid_char for char in name):
        messagebox.showerror("Error", "Invalid characters: \n" + str(string.punctuation))
    else:
   
        # 1. Remove all whitespace at the start &  end
        # 2. Whitespace between characters with length > 1 are set with length 1
        name = ' '.join(name.split())

        # name with length zero
        # input name is all whitespaces
        if not name and which == "first":
            messagebox.showinfo("Info", "Please enter your first name")
            
        elif not name and which == "last":
            messagebox.showinfo("Info", "Please enter your last name")

        else:
            return name


        



def register():
    '''
    Register a member when the `Registration` button is pressed
    '''
    print("\n##### Start Registration #####")

    global member_num

    first_name = entry1.get()
    last_name = entry2.get()


    # Check if the user input the name correctly
    first_name = check_name(first_name, "first")
    if first_name == None:
        entry1.delete(0, 'end')
        entry2.delete(0, 'end')
        print("Invalid Input for first name")
        return -1
    
    last_name = check_name(last_name, "last")
    if last_name == None:
        entry1.delete(0, 'end')
        entry2.delete(0, 'end')
        print("Invalid Input for last name")
        return -1

    entry1.delete(0, 'end')
    entry2.delete(0, 'end')


    full_name = first_name + "_" + last_name
    full_name = full_name.replace(" ", "_")
    file_path = "dataset/" + full_name




    # Check if the full_name has been registered or not
    if not os.path.exists(file_path):
        msg_box = messagebox.askquestion('Registration', 
                'Your name for member registration is \n' 
                + '"'+ first_name + " " + last_name + '"' + '  Is it correct?\n\nIf yes, photos would be taken for registration', icon = 'info')

        if msg_box == 'yes':
            # Create a directory to store the photos
            os.makedirs(file_path)

            # Take photos now!
            messagebox.showinfo("How to take photos", "Press SPACEBAR to take photo\n (Max: 10 photos)\n\nTry to have different gesture / orientation when taking photos")
            
            img_num = take_photos.take_photos(file_path)

            if img_num <= 0:
                messagebox.showerror("Error", "No photos are taken!\n" + '"' + first_name + "  " + last_name + '"' + " is NOT registered as a member")
                os.rmdir(file_path)
            
            else:

                # Extract 128-d feature vectors from the new images
                # and append to the pickle files
                image_to_vector.gen_vector_register(full_name, 0.3)

                # Train SVM from all 128-d vectors
                training.svm()

                # Train KNN from all 128-d vectors
                training.knn()


                # Train Random Forest from all 128-d vectors
                training.rf()


                # Update the number of registered member
                member_num += 1
                label.config(text = "Total number of member: " + str(member_num))
                
                print("##### Successful registration ######\n")

    else:
        messagebox.showerror("Error", "This name has been registered")
        print("Duplicated registration")
        print("Registration rejected\n")



def add_photos():
    '''
    Add photos to an existing member when the `Add photos` button is pressed
    '''
    print("\n##### Adding photos #####")

    first_name = entry1.get()
    last_name = entry2.get()

    # Check if the user input the name correctly
    first_name = check_name(first_name, "first")
    if first_name == None:
        entry1.delete(0, 'end')
        entry2.delete(0, 'end')
        print("Invalid Input for first name")
        return -1
    
    last_name = check_name(last_name, "last")
    if last_name == None:
        entry1.delete(0, 'end')
        entry2.delete(0, 'end')
        print("Invalid Input for last name")
        return -1

    entry1.delete(0, 'end')
    entry2.delete(0, 'end')

    full_name = first_name + "_" + last_name
    full_name = full_name.replace(" ", "_")
    file_path = "dataset/" + full_name


    # Check if the full_name has been registered or not
    if os.path.exists(file_path):
        msg_box = messagebox.askquestion('Add photos', 
                'The member to add photos is \n' 
                + '"'+ first_name + " " + last_name + '"' + '  Is it correct?\n\n', icon = 'info')

        if msg_box == 'yes':
            # Create a directory to store the photos

            # Take photos now!
            messagebox.showinfo("How to take photos", "Press SPACEBAR to take photo\n (Max: 10 photos)\n\nTry to have different gesture / orientation when taking photos")
            
            img_num = take_photos.take_photos(file_path)

            if img_num <= 0:
                messagebox.showerror("Error", "No photos are added to " + '"' + first_name + "  " + last_name + '"')
            
            else:

                # Extract 128-d feature vectors from the new images
                # and append to the pickle files
                image_to_vector.gen_vector_add(full_name, 0.3)

                # Train SVM from all 128-d vectors
                training.svm()

                # Train KNN from all 128-d vectors
                training.knn()

                # Train Random Forest from all 128-d vectors
                training.rf()


                print("##### Photos Added ######\n")

    else:
        messagebox.showerror("Error", "This name has not yet registered")
        print("No photos are added")







if __name__ == "__main__":
    
    # Create the GUI window
    window = tk.Tk()
    window.title("Face Recognition Demo")
    window.geometry('600x380+250+200')

    # Number of member
    member_num = 0

    # Initialization
    initialization()

    # First name input bar
    tk.Label(window, text = "First Name").grid(row = 0, column = 0)
    entry1 = tk.Entry(window)
    entry1.grid(row = 0, column = 1)

    # Last name input bar
    tk.Label(window, text = "Last Name").grid(row = 1, column = 0)
    entry2 = tk.Entry(window)
    entry2.grid(row = 1, column = 1)

    # Show the total number of member
    label = tk.Label(window)
    label.config(text = "Total number of member: " + str(member_num))
    label.grid(row = 2, column = 1, sticky=tk.W, pady=4)


    # Buttons to perform various tasks as stated in the `text`
    tk.Button(window, text='Registration', command=register).grid(row=2, column=0, sticky=tk.W, pady=4)
    tk.Button(window, text='Add photos', command=add_photos).grid(row=4, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (Support Vector Machine)', command=face_recognition_svm).grid(row=7, column=0, sticky=tk.W, pady=4)
    tk.Button(window,text='Face Recognition (k-Nearest Neighbours)', command=face_recognition_knn).grid(row=8, column=0, sticky=tk.W, pady=4)
    tk.Button(window,text='Face Recognition (Random Forest)', command=face_recognition_rf).grid(row=9, column=0, sticky=tk.W, pady=4)
    tk.Button(window,text='Face Recognition (Pearson Correlation)', command=face_recognition_pearson).grid(row=11, column=0, sticky=tk.W, pady=4)
    tk.Button(window,text='Face Recognition (Cosine Similarity)', command=face_recognition_cosine).grid(row=12, column=0, sticky=tk.W, pady=4)
    tk.Button(window,text='Face Recognition (L2 Distance)', command=face_recognition_l2).grid(row=13, column=0, sticky=tk.W, pady=4)
    tk.Button(window,text='Face Recognition (L1 Distance)', command=face_recognition_l1).grid(row=14, column=0, sticky=tk.W, pady=4)


    tk.Button(window, text='Information', command=information).grid(row=7, column=3, sticky=tk.E, pady=4)
    tk.Button(window, text='Restart', command=re_initialization).grid(row=8, column=3, sticky=tk.E, pady=4)
    tk.Button(window, text='Quit', command=window.quit).grid(row=9, column=3, sticky=tk.E, pady=4)

    window.mainloop()



