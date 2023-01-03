import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from tqdm import tqdm

from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense
from tensorflow.keras.models import Sequential

from tensorflow import keras
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
from tensorflow.keras.utils import plot_model

IMG_SIZE = (500,500)
TRAIN_DIR = '/content/drive/MyDrive/Fovea/dataset/AE/Train/'
VAL_DIR = '/content/drive/MyDrive/Fovea/dataset/AE/Test/'

def load_data(dir_path, img_size=(100,100)):
   
    X = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file, 0) #turning grayscale
                    img = cv2.resize(img,IMG_SIZE)
                    X.append(img)
            i += 1
    X = np.array(X)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X

x = load_data(TRAIN_DIR, IMG_SIZE)
y = load_data(VAL_DIR, IMG_SIZE)

count=5
plt.figure(figsize=(20,10))

for i in range(count):
    ax = plt.subplot(2, count, i + 1)
    plt.imshow(x[i].reshape(500, 500))
    
    plt.gray()
    
plt.show()

x=x.reshape((len(x),np.prod(x.shape[1:])))
y=y.reshape((len(y),np.prod(y.shape[1:])))

#normalization
x = x.astype('float32') / 255.
y = y.astype('float32') / 255.

print(x.shape)
print(y.shape)

#whole autoencoder
input_l=Input(shape=(250000,))

encoding_1=Dense(512, activation='relu')(input_l)
encoding_2=Dense(256, activation='relu')(encoding_1)

bottleneck=Dense(128, activation='relu')(encoding_2)

decoding_1=Dense(256, activation='relu')(bottleneck)
decoding_2=Dense(512, activation='relu')(decoding_1)

output_l=Dense(250000, activation='tanh')(decoding_2)

autoencoder=Model(inputs=[input_l],outputs=[output_l])

autoencoder.summary()

plot_model(autoencoder, to_file='autoencoder_summarized.png', show_shapes=True, show_layer_names=True)

#Just the encoder, useful to extract bottleneck representation
encoder=Model(inputs=[input_l],outputs=[bottleneck])

encoder.summary()

plot_model(encoder, to_file='encoder_summarized.png', show_shapes=True, show_layer_names=True)

#reconstructed image
encoded_input=Input(shape=(128,))

decoded_layer_1=autoencoder.layers[-3](encoded_input)
decoded_layer_2=autoencoder.layers[-2](decoded_layer_1)

decoded=autoencoder.layers[-1](decoded_layer_2)
decoder=Model(inputs=[encoded_input],outputs=[decoded])

decoder.summary()

plot_model(decoder, to_file='model_plot_decoder.png', show_shapes=True, show_layer_names=True)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x, x,
                epochs=5,
                batch_size=6,
                shuffle=True,
                validation_data=(y, y))

encoded= encoder.predict(x)
decoded = decoder.predict(encoded)

predicted_images=autoencoder.predict(x)

n = 10 
plt.figure(figsize=(20, 4))
for i in range(n):
    
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x[i].reshape((500), 500))
    plt.gray()

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(predicted_images[i].reshape(500, 500))
    plt.gray()
    
plt.show()

n = 10 
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(decoded[i].reshape(500, 500))
    plt.gray()
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded[i].reshape(32,4))
    plt.gray()
    
plt.show()

print(encoded[0])
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

# print(encoded)

autoencoder.save('autoencoder_1_128.h5')
encoder.save('encoder_1_128.h5')
decoder.save('decoder_1_128.h5')