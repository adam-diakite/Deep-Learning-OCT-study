import cv2
import numpy as np #array management
import os
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix #performance calculation
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
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

IMG_SIZE = (500,500)

def load_data(dir_path, img_size=(100,100)):
	X = []
	i = 0
	files = []
	labels = dict()
	for file in os.listdir(dir_path):
		if not file.startswith('.'):
			img = cv2.imread(dir_path + '/' + file, 0) #turning grayscale
			img = cv2.resize(img,IMG_SIZE)
			X.append(img)
			i = i+1
			files.append(os.path.splitext(file)[0])
			print(i)
	X = np.array(X)
	print(f'{len(X)} images loaded from {dir_path} directory.')
	return { "x": X, "filenames":files }

encoder = load_model('/hps/nobackup/birney/projects/ukb_eye_image_analysis/resources/encoder_1_128.h5')
decoder = load_model('/hps/nobackup/birney/projects/ukb_eye_image_analysis/resources/decoder_1_128.h5')
outputdir = "/hps/nobackup/birney/projects/ukb_eye_image_analysis/encoder_output"

data = load_data("/hps/nobackup/birney/projects/ukb_eye_image_analysis/central_slices/", IMG_SIZE)
x = data['x']
x=x.reshape((len(x),np.prod(x.shape[1:])))
#normalization
x = x.astype('float32') / 255.
prd = encoder.predict(x)
dec = decoder.predict(prd)

for index in range(0, len(prd)):
	txt_outfilename = outputdir + "/" + data['filenames'][index] + ".txt"
	png_outfilename = outputdir + "/" + data['filenames'][index] + ".png"
	np.savetxt(txt_outfilename, prd[index], fmt='%f')
	plt.figure(figsize=(20,20))
	ax = plt.subplot(1, 3, 1)
	plt.imshow(x[index].reshape(500, 500))
	ax = plt.subplot(1, 3, 2)
	plt.imshow(dec[index].reshape(500, 500))
	ax = plt.subplot(1, 3, 3)
	plt.imshow(prd[index].reshape(8, 16))
	plt.savefig(png_outfilename)