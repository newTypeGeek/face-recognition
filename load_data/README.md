# load_data
`load_data` package consists of functions to load ML models and 128-d vectors to memory


1. `load_clf.py`<br>
   Load trained classifier models
   1. `load_svm`: load *support vector machine*  model
   2. `load_knn`: load *k-nearest neighbours* model
   3. `load_rf`: load *random forest* model

2. `load_cnn.py`<br>
   Load pre-trained CNN models
   1. `resnet`: load ResNet + SSD model for face detection
   2. `facenet`: load FaceNet model for embedding

3. `load_vec.py`<br>
   Load 128-d vectors (embeddings)
   1. `vector`: load 128-d vectors, and the corresponding identities
