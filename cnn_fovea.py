import numpy as np #array management
from tqdm import tqdm
import cv2 #image processing
import os #directory navigator
import shutil #high level file operations
import itertools #creating iterators
import imutils #image processing (rotation, etc)
import plotly.graph_objs as go #beautiful graphs


import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix #performance calculation

from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.preprocessing.image import ImageDataGenerator #data aug
from keras.applications.vgg16 import VGG16, preprocess_input

from keras import layers #model construction 
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam #optimizer
from tensorflow.keras.optimizers import RMSprop #optimizing the gradient descent 

from keras.callbacks import EarlyStopping #avoid overfitting 

RANDOM_SEED = 123

TRAIN_DIR = '/Users/adamdiakite/Desktop/Fovea/dataset/CNN/Train/'
TEST_DIR = '/Users/adamdiakite/Desktop/Fovea/dataset/CNN/Test/'
VAL_DIR = '/Users/adamdiakite/Desktop/Fovea/dataset/CNN/Val/'
IMG_SIZE = (224,224)

def load_data(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels



# use predefined function to load the image data into workspace
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)



#Image cropping for more precision - borders are not interesting

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

X_train_prep = preprocess_imgs(set_name=X_train, img_size=IMG_SIZE)
X_test_prep = preprocess_imgs(set_name=X_test, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val, img_size=IMG_SIZE)


train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary', #type of labels - could be categorical
    seed=RANDOM_SEED
)


validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)

# load base model
vgg16_weight_path = '/Users/adamdiakite/Desktop/Fovea/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)


EPOCHS = 10
es = EarlyStopping(
    monitor='accuracy', 
    mode='max',
    patience=6
)

history = model.fit(
    train_generator,      
    steps_per_epoch=len(X_train)//32,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(X_test)//16,
    callbacks=[es]
)

model.save('CNN_fovea.h5')

