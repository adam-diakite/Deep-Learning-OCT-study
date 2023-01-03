import numpy as np #array management
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm 
import random 
from PIL import Image
from keras import layers #model construction
from keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from skimage.io import imread, imshow


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (128, 128)
IMG_CHANNELS = 1

#Paths
RESULT_PATH = "/Users/adamdiakite/Desktop/OCTLine-Seg/21017_1_0/"
model = load_model('/Users/adamdiakite/Desktop/OCTLine-Seg/model-saves/u-net-fovea.h5')

def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[1])

def load_data(dir_path, img_size=(128,128)):
   
    X = []
    patient_number = []

    i = 0
    for path in tqdm(sorted(os.listdir(dir_path), key = extract_integer )):
        if not path.startswith('.'):
            img = cv2.imread(dir_path + path  , 0) #turning grayscale
            img = cv2.resize(img,IMG_SIZE)
            X.append(img)
        i += 1
    X = np.array(X)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X

def load_image_name(dir_path):
   
    X = []
    patient_number = []

    i = 0
    for path in tqdm(sorted(os.listdir(dir_path), key = extract_integer )):
        if not path.startswith('.'):
            X.append(path)
        i += 1
    X = np.array(X)
    print(f'{len(X)} names loaded from {dir_path} directory.')
    return X

slice_name = load_image_name(RESULT_PATH)
images = load_data(RESULT_PATH, IMG_SIZE)

line_ex = model.predict(images, verbose=1)
lines = (line_ex > 0.5).astype(np.uint8)
lines = lines.squeeze()



def show_scans(cols, rows, image = []):
    w = 16
    h = 8
    fig = plt.figure(figsize=(cols, cols))
    for i in range (1, cols*rows - 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(image[i])
    return plt.show()

show_scans(16, 8, images)
show_scans(16, 8, lines)

#calculates the distance between two white points for each y axis of the image
def calc_dist(array = []):
    columns, rows = array.shape
    array = array.T
    line_distance = []

    for c in range (columns):        
        distance = 0
        for r in range(rows):
            if array[c][r] != 1:
                distance += 1 
        line_distance.append(len(array) - distance)
    return line_distance


#Adding the image number to find the right slice
for i in range (len(lines)):
    results = calc_dist(lines[i])
    results.insert(0, slice_name[i])
    print (results)
