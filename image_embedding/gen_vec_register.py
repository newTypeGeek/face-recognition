#!/usr/bin/env python3

#######################
# gen_vec_register.py #
#######################

import os
import sys
import shutil
import numpy as np

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from image_embedding.img_to_vec import img_to_vec
from image_embedding.storage import serialize
from image_embedding.storage import deserialize
from load_data import load_cnn


def gen_vec_register(identity, min_prob_filter):
    '''
    For registration
    
    0. Loops all images in the ../dataset/<identity>,
    1. Convert them to 128-d vectors,
    2. Save the serialized dictionary (image_paths, 128-d vectors)
       to path:  ../embeddings/data/individual/<identity>/embeddings.pickle

    Arguments:
    1. identity:            Name for registration

    2. min_prob_filter:     Probability threshold to filter weak detection by ResNet

    '''
    print("##########################")
    print("##########################")
    print("#####   New vector   #####")
    print("#####   New member   #####")
    print("##########################")
    print("##########################\n")

    DATASET_PATH = "dataset"
    EMBEDDING_OUTPUT = "embeddings/data"
    FILE_NAME = "embeddings.pickle"
    INDIVIDUAL = "individual"
    OVERALL = "overall"


    # Load the CNN for face detection and 128-d vector extraction
    detector = load_cnn.resnet()
    embedder = load_cnn.facenet()

    print("\n*** Processing images with identity: " + identity + " ***")
        
    # Initialize list to store image path and 128-d vector
    # for THIS identity
    individual_image_paths = []
    individual_vectors = []

    # Total number of vector for THIS identity
    vector_num = 0

    # Setting up image path
    identity_path = os.path.join(DATASET_PATH, identity)
    image_files = [x for x in os.listdir(identity_path) if not x.startswith('.')]
    image_num = len(image_files)

    # Create directory for THIS identity to store 128-d vector
    vector_path = os.path.join(EMBEDDING_OUTPUT, INDIVIDUAL, identity)
    if not os.path.isdir(vector_path):
        os.makedirs(vector_path)
    else:
        print("WARNING: The path " + vector_path + " exists")
        print("         Removing all its contents")
        shutil.rmtree(vector_path)
        os.makedirs(vector_path)
    


    # Loop over images for THIS identity
    for image_file in image_files: 
        image_path = os.path.join(identity_path, image_file)

        # Extract 128-d vector from the image
        vector = img_to_vec(image_path, detector, embedder, min_prob_filter)

        # Skip below if vector is not extracted
        if np.isfinite(vector).all() == False:
            print("WARNING: No 128-d vector extracted for " + image_path)
            continue                

        # For individual member
        individual_image_paths.append(image_path)
        individual_vectors.append(vector)

        vector_num += 1

    # Save the extracted 128-d vectors for THIS identity
    print("Serializing {} 128-d vectors for {} images".format(vector_num, image_num))
    output_path = os.path.join(EMBEDDING_OUTPUT, INDIVIDUAL, identity, FILE_NAME)
    data_new = dict(zip(individual_image_paths, individual_vectors))
    serialize(data_new, output_path)

    # NOTE: Do not forget to append the data for the overall file
    output_path = os.path.join(EMBEDDING_OUTPUT, OVERALL, FILE_NAME)
    data_all = deserialize(output_path)
    data_all.update(data_new)
    serialize(data_all, output_path)
    print("*** Done with identity: " + identity + " ***\n")

