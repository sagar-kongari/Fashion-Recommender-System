# file handling
import os
import pickle
from tqdm import tqdm

# array and normalization
import numpy as np
from numpy.linalg import norm

# tensorflow and keras dependency
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# transfer learning model import
model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
model.trainable = False

# building model
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# function to convert image to numeric matrix array
def feature_array(pic_path, model):
    pic = image.load_img(pic_path, target_size=(32,32))
    pic_array = image.img_to_array(pic)
    expand_pic_array = np.expand_dims(pic_array, axis=0)
    final_pic = preprocess_input(expand_pic_array)
    result = model.predict(final_pic).flatten()
    norm_result = result/norm(result)
    return norm_result

# accessing filepath of images and store in a list
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# New list to store extracted features from every image
featurelist = []
for file in tqdm(filenames):
    featurelist.append(feature_array(file, model))

# saving our lists as a pickle file
pickle.dump(featurelist, open('features.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))