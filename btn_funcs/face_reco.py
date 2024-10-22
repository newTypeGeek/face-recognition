from tkinter import messagebox

import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from webcam.face_reco_video import face_reco_video


def l1_norm(member):
    '''
    Perform face recognition using L1 distance from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "l1")    
    else:
        messagebox.showerror("Error", "No registered members")


def l2_norm(member):
    '''
    Perform face recognition using L2 distance from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "l2")
    else:
        messagebox.showerror("Error", "No registered members")


def cosine_sim(member):
    '''
    Perform face recognition using cosine similarity from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "cosine")
    else:
        messagebox.showerror("Error", "No registered members")


def pearson(member):
    '''
    Perform face recognition using Pearson correlation from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "pearson")
    else:
        messagebox.showerror("Error", "No registered members")


def svm(member):
    '''
    Perform face recognition using SVM from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "svm")
    else:
        messagebox.showerror("Error", "No registered members")


def knn(member):
    '''
    Perform face recognition using k-NN from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "knn")
    else:
        messagebox.showerror("Error", "No registered members")


def rand_forest(member):
    '''
    Perform face recognition using random forest from 128-d vectors
    '''
    if member.num > 0:
        face_reco_video(0.8, "rf")
    else:
        messagebox.showerror("Error", "No registered members")

