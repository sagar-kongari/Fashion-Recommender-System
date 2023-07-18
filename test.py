# computer vision to work with images
import cv2

# to load saved model files 
import pickle

# linear algebra
import numpy as np
from numpy.linalg import norm

# tensorflow and keras dependency
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

# KNN algorithm
from sklearn.neighbors import NearestNeighbors

# uploading model
featurelist = pickle.load(open('features.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# transfer model import
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

# building model
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# convert image to numeric matrix array
pic = image.load_img('sample.jpg', target_size=(224,224))
pic_array = image.img_to_array(pic)
expand_pic_array = np.expand_dims(pic_array, axis=0)
final_pic = preprocess_input(expand_pic_array)
result = model.predict(final_pic).flatten()
norm_result = result/norm(result)

# classification algorithm
classifier = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
classifier.fit(featurelist)

distance, indices = classifier.kneighbors([norm_result])

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', temp_img)
    cv2.waitKey(0)