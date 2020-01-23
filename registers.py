import os
import string
from tkinter import messagebox

import training
import take_photos
import image_to_vector


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



def register(member, entry1, entry2, label):
    '''
    Register a member when the `Registration` button is pressed
    '''
    print("\n##### Start Registration #####")

    member_num = member.num

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
                member.num = member_num
                print("##### Successful registration ######\n")

    else:
        messagebox.showerror("Error", "This name has been registered")
        print("Duplicated registration")
        print("Registration rejected\n")




def add_photos(entry1, entry2):
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


