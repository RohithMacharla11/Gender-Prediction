import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import kagglehub
from tqdm import tqdm

def load_dataset():
    # Download dataset
    path = kagglehub.dataset_download("jangedoo/utkface-new")
    dir_path = path + "/UTKFace/"
    
    image_paths = []
    gender_labels = []
    
    for filename in tqdm(os.listdir(dir_path)):
        image_path = os.path.join(dir_path, filename)
        temp = filename.split('_')
        gender_labels.append(int(temp[1]))
        image_paths.append(image_path)
    
    df = pd.DataFrame()
    df['image'] = image_paths
    df['gender'] = gender_labels
    return df

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)
    return img_array