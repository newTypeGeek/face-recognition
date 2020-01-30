# image_embedding
`image_embedding` package consists of functions to convert photos to 128-d vectors


1. `gen_vec_start.py`<br>
   1. `gen_vec_start` is triggered when the program just starts or restarts from GUI. It clears all the saved 128-d vectors in storage, re-scan all photos from `dataset` directory and convert them to 128-d vectors, and save them.

2. `gen_vec_register.py`<br>
   1. `gen_vec_register` is triggered when a new member is registered. It scans only the photos of the new member, convert to 128-d vectors and save them.

3. `gen_vec_add.py`<br>
   1. `gen_vec_add` is triggered when new photos are added to an existing member. Convert only the new photos to 128-d vectors and save them.

4. `img_to_vec.py`<br>
   1. `img_to_vec` is used only by the above three functions. For a single photo, it performs face detection followed by embedding (convert to a 128-d vector)

5. `storage.py`<br>
   1. `deserialize` deserialize the pickle file to an object (dictionary type with *image path* as key and *128-d numpy array* as value)

   2. `serialize` serializes the dictionary to a pickle file
