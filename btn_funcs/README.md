# btn_funcs
`btn_funcs` package (known as button functions) consists of functions to perform various tasks in the main program (`main.py`)

1. `control.py`<br>
   1. `info`: display references in GUI
   2. `start`: initialize the program, including training ML models
   3. `restart`: restart the program from GUI, including training ML models


2. `face_reco.py`<br>
   Perform face recognition (i.e. classification of 128-d vectors)
   1. `l1_norm`: L1 norm
   2. `l2_norm`: L2 norm
   3. `cosine_sim`: Cosine similarity
   4. `pearson`: Pearson correlation
   5. `svm`: Support vector machine
   6. `knn`: k-nearest neighbours
   7. `rand_forest`: Random forest


3. `registers.py`<br>
    Perform actions related to members
    1. `check_name`: verify the input name from GUI has no special characters
    2. `register`: register a new member
    3. `add_photos`: add more photos for a specified existing member 
