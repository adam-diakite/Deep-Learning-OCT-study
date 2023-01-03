#re-importing everything to avoid problems
import numpy as np #array management
import os
from tqdm import tqdm
import cv2 #image processing
import os #directory navigator
import shutil #high level file operations
import itertools #creating iterators
import imutils #image processing (rotation, etc)
import matplotlib.pyplot as plt 
import cv2 
import numpy as np
from PIL import Image

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix #performance calculation

import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.preprocessing.image import ImageDataGenerator #data aug
from keras.applications.vgg16 import VGG16, preprocess_input

from keras import layers #model construction 
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam #optimizer
from tensorflow.keras.optimizers import RMSprop #optimizing the gradient descent 


from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image

from keras.callbacks import EarlyStopping #avoid overfitting 


model = load_model('/Users/adamdiakite/Desktop/Fovea/CNN_fovea.h5')

#integer at the end of filename to sort better
def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[1])

#test 
#print(sorted(os.listdir("/Users/adamdiakite/Downloads/03/67/1000367/21017_0_0"), key=extract_integer)) # returns list

images = []
for img in sorted(os.listdir('/Users/adamdiakite/Downloads/1000154-2/21018_1_0'), key=extract_integer):
    img = os.path.join('/Users/adamdiakite/Downloads/1000154-2/21018_1_0', img)
    #resizing the image
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

#stacking vertically for display
images = np.vstack(images)
classes = model.predict(images)

print(classes)

#Indexes of the max value and the min value

print ('Minimum score index is : ', classes.argmin(axis=0))
print ('Maximum score index is : ', classes.argmax(axis=0))


