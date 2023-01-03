import os
import cv2
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np
import random 
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from warnings import filterwarnings

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (128, 128)
IMG_CHANNELS = 1

TRAIN_PATH = '/Users/adamdiakite/Desktop/OCTLine-Seg/grayscale-train/'
MASK_PATH = '/Users/adamdiakite/Desktop/OCTLine-Seg/grayscale-masks/'
TEST_PATH = '/Users/adamdiakite/Desktop/OCTLine-Seg/grayscale-test/'

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

train_ids = [f for f in listdir(TRAIN_PATH) if isfile(join(TRAIN_PATH, f))]
mask_ids = [f for f in listdir(MASK_PATH) if isfile(join(MASK_PATH, f))]
test_ids = [f for f in listdir(TEST_PATH) if isfile(join(TEST_PATH, f))]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)


# print('Resizing masks images')
# for n, id_ in tqdm(enumerate(mask_ids), total=len(mask_ids)): 

#     mask_path = MASK_PATH 
#     mask = imread(mask_path  + id_ , plugin='matplotlib')[:,:,:IMG_CHANNELS]  
#     mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     Y_train[n] = mask  #Fill empty Y_train with values from mask

# #resizing images and append into np arrays
# print('Resizing training images')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): 

#     path = TRAIN_PATH     
#     img = imread(path  + id_ , plugin='matplotlib')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train[n] = img  #Fill empty X_train with values from img


# # test images processing
# print('Resizing test images') 
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#     path = TEST_PATH
#     img_mask = imread(path + id_ , plugin='matplotlib') [:,:,:IMG_CHANNELS]  
#     img_mask = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_test[n] = img_mask

def load_data(dir_path, img_size=(128,128)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    i = 0
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            
            img = cv2.imread(dir_path + path  , 0) #turning grayscale
            img = cv2.resize(img,IMG_SIZE)
            X.append(img)
        i += 1
    X = np.array(X)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X

X_train = load_data(TRAIN_PATH, IMG_SIZE)
Y_train = load_data(MASK_PATH, IMG_SIZE)
X_test = load_data(TEST_PATH, IMG_SIZE)

#Normalizing masks
Y_train = Y_train/255
Y_train = (Y_train > .5).astype(int)

#Shwoing example of train image and train mask
image_x = random.randint(0, len(train_ids ) - 1 )
imshow(X_train[image_x])
plt.show()

imshow(np.squeeze(Y_train[image_x]))
plt.show()
print('Finally done')

####################################

#model building
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Encoder path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Decoder path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Checkpoint
chekpointer = tf.keras.callbacks.ModelCheckpoint('model_for_segmentation_ex .h5', verbose = 1, save_best_only = True)
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience = 2, monitor ='val_loss'),
#     tf.keras.callbacks.TensorBoard(log_dir='logs')]

#model fit + save
results = model.fit(X_train, Y_train, validation_split = 0.1, batch_size = 4, epochs = 50)
model.save('/Users/adamdiakite/Desktop/OCTLine-Seg/model-saves/line-extractor.h5')

####################################

idx = random.randint(0, len(X_train))

#Now, each pixel has a probability of being in one of the two classes. (line or not)
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

#For the predicting, binary classification so the thresh is at 0.5
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.bool8)

ix = random.randint(0, len(preds_train_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(preds_test_t[ix].astype(int)))
plt.show()


# #Perform a sanity check on some random training samples
# ix = random.randint(0, len(preds_train_t))
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(Y_train[ix].astype(int)))
# plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.show()

# Perform a sanity check on some random validation samples
# ix = random.randint(0, len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# plt.show()
# imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix].astype(int)))
# plt.show()
# imshow(np.squeeze(preds_val_t[ix].astype(int)))
# plt.show()

