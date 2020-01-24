#!/usr/bin/env python3

#############
#  main.py  #
#############

# Uncomment these lines if you are planning to 
# build a standalone executable using PyInstaller
# import multiprocessing
# multiprocessing.freeze_support()

import tkinter as tk
from functools import partial
from btn_funcs import control, registers, face_reco


class Member:
    '''
    Use to count the total number of registered member.
    It is a workaround for setting global variable
    '''
    def __init__(self):
        self.num = 0


if __name__ == "__main__": 
    # Create the GUI window
    window = tk.Tk()
    window.title("Face Recognition Demo")
    window.geometry('600x380+250+200')

    # Create member object,
    # just to store number of registered member
    member = Member()

    # Initialization
    control.start(member)

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
    label.config(text = "Total number of member: " + str(member.num))
    label.grid(row = 2, column = 1, sticky=tk.W, pady=4)


    # Buttons to perform various tasks as stated in the `text`
    tk.Button(window, text='Registration',
              command=partial(registers.register, member, entry1, entry2, label)).grid(row=2, column=0, sticky=tk.W, pady=4)

    tk.Button(window, text='Add photos',
              command=partial(registers.add_photos, entry1, entry2)).grid(row=4, column=0, sticky=tk.W, pady=4)


    tk.Button(window,text='Face Recognition (Support Vector Machine)',
              command=partial(face_reco.svm, member)).grid(row=7, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (k-Nearest Neighbours)',
              command=partial(face_reco.knn, member)).grid(row=8, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (Random Forest)',
              command=partial(face_reco.rand_forest, member)).grid(row=9, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (Pearson Correlation)',
              command=partial(face_reco.pearson, member)).grid(row=11, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (Cosine Similarity)',
              command=partial(face_reco.cosine_sim, member)).grid(row=12, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (L2 Distance)',
              command=partial(face_reco.l2_norm, member)).grid(row=13, column=0, sticky=tk.W, pady=4)

    tk.Button(window,text='Face Recognition (L1 Distance)',
              command=partial(face_reco.l1_norm, member)).grid(row=14, column=0, sticky=tk.W, pady=4)


    tk.Button(window, text='Information',
              command=control.info).grid(row=7, column=3, sticky=tk.E, pady=4)

    tk.Button(window, text='Restart',
              command=partial(control.restart, member, label)).grid(row=8, column=3, sticky=tk.E, pady=4)

    tk.Button(window, text='Quit',
              command=window.quit).grid(row=9, column=3, sticky=tk.E, pady=4)


    window.mainloop()
