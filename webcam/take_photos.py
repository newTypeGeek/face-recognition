#!/usr/bin/env python3

##################
# take_photos.py #
##################


import cv2
import numpy as np
from imutils.video import VideoStream
from imutils import resize
import os


def take_photos(photos_path):
    '''
    Take photos from a camera. 
    User is required to press SPACEBAR to take photos

    Arguments:
    1. photos_path: the path to store the photos


    Returns:
    1. img_num:     total number of photos taken
    '''
    

    # NOTE: To handle non-consecutive image naming
    # For example, the current image file names might be 
    # ['0001.jpg', '0004.jpg', '0006.jpg, 0007.jpg']
    # 
    # When taking photos, the new naming should first use this set
    # 0000.jpg, 0002.jpg, 0003.jpg, 0005.jpg
    #
    # then, followed by
    # 0008.jpg, 0009.jpg, 0010.jpg ... and so on
    photos_exist = [x for x in os.listdir(photos_path) if not x.startswith('.')]
    num_photos_exist = len(photos_exist)
    idx_required = []
    idx_max = 0


    # Check if there are photos in photos_path
    if num_photos_exist > 0:
        # Convert all photos name to integer value
        idx = [ 0 if photo == "0000.jpg" else int(os.path.splitext(photo)[0].lstrip('0')) for photo in photos_exist ]
        idx_max = max(idx)
        
        # Find the missing integer between 0 to n (inclusive)
        # These indices would be used for saving the image first
        all_idx = list(range(0, idx_max+1))
        idx_required = sorted( list(set(all_idx).difference(set(idx))) )
        idx_max = idx_max + 1





    frame_name = "Photo Taking"
    
    cam = VideoStream(0).start()

    cv2.namedWindow(frame_name)
    cv2.moveWindow(frame_name, 500,200);

    img_num = 0

    while True:
        frame = cam.read()
        if frame.any() == None:
            break

        frame = resize(frame, width=1024)
        frame = cv2.flip(frame, 1)

        frame_with_text = np.copy(frame)
        
        text = "Number of photos taken = " + str(img_num) + "/10"
        cv2.putText(frame_with_text, text, (20, 450),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 2)

        text = "Press SPACEBAR to take photo"
        cv2.putText(frame_with_text, text, (20, 500),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)

        text = "Press ESC to Exit"
        cv2.putText(frame_with_text, text, (20, 550),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 128, 0), 2)



        cv2.imshow(frame_name, frame_with_text)

        # TODO: Handle unicode input
        #       Now, if the keyboard is Chinese or Russian
        #       and when the keystroke (QWER ..etc) is pressed
        #       the program stops with assertion error
        # k = cv2.waitKey(1)
        k = cv2.waitKeyEx(1)

        # ESC is pressed or 10 photos are taken
        if k % 256 == 27 or img_num >= 10:
            break

        # SPACEBAR is pressed to take a photo
        elif k % 256 == 32:

            if len(idx_required) > 0:
                name = str(idx_required.pop(0)).zfill(4)
            else:
                name = str(idx_max).zfill(4)
                idx_max += 1
            
            img_name = os.path.join(photos_path, name + ".jpg")
            cv2.imwrite(img_name, frame) 
            img_num += 1
            print("Image saved to " + img_name)



    cam.stop()
    cv2.destroyAllWindows()

    return img_num
