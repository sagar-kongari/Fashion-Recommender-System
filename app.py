# app 
import streamlit as st

# working with different types of image
from PIL import Image

import os
import pickle
import numpy as np
from numpy.linalg import norm

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

from sklearn.neighbors import NearestNeighbors

featurelist = np.array(pickle.load(open('features.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# title of webapp page
st.title('Style Recommender System')

# function to save upload image file
def upload(upload_file):
    try:
        with open(os.path.join('uploads', upload_file.name), 'wb') as f:
            f.write(upload_file.getbuffer())
        return 1
    except:
        return 0

# function to convert image to numeric matrix array
def feature_array(pic_path, model):
    pic = image.load_img(pic_path, target_size=(224,224))
    pic_array = image.img_to_array(pic)
    expand_pic_array = np.expand_dims(pic_array, axis=0)
    final_pic = preprocess_input(expand_pic_array)
    result = model.predict(final_pic).flatten()
    norm_result = result/norm(result)
    return norm_result

# function to recommend using knn algorithm
def recommend(features,feature_list):
    classifier = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    classifier.fit(feature_list)
    indices = classifier.kneighbors([features])
    return indices

upload_file = st.file_uploader('upload an image')
if upload_file is not None:
    if upload(upload_file):
        display_image = Image.open(upload_file)
        st.image(display_image)
        features = feature_array(os.path.join('uploads', upload_file.name), model)
        indices = recommend(features, featurelist)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("file upload error")