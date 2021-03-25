from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
from PIL import ImageFile

from keras.preprocessing import image

from hotdog_classifiers.models import define_VGG16_fc2_out


def save_feature(save_path, feature):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature)


def feature_extract_vgg16(files_dir, features_dir):
    model_vgg16_fc2 = define_VGG16_fc2_out()

    files_names = os.listdir(files_dir)
    files_names.sort() # 0000 .... 0001 ...

    for file_name_with_extension in files_names:
        file_name = os.path.splitext(file_name_with_extension)[0]

        img_path = os.path.join(files_dir, file_name_with_extension)

        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)

        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        print("[+] Extract feature from image : ", img_path)

        feature = model_vgg16_fc2.predict(img_data)

        save_path = os.path.join(features_dir, f'{file_name}.npy')

        save_feature(save_path, feature)


def load_npy_from_dir(npy_files_dir):
    numpy_arrays = [] 
    npy_files_names = os.listdir(npy_files_dir)
    npy_files_names.sort()
    for npy_file_name in npy_files_names:
        npy_file_name_path = os.path.join(npy_files_dir, npy_file_name)
        npy_array = np.load(npy_file_name_path)
        numpy_arrays.append(npy_array)
    return numpy_arrays
