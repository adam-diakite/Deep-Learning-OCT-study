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

model = load_model('/hps/nobackup/birney/projects/ukb_eye_image_analysis/resources/CNN_fovea.h5')

def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[1])

def fit_model(dir_path):
	images = []
	for img in sorted(os.listdir(dir_path), key=extract_integer):
	    img = os.path.join(dir_path, img)
	    #resizing the image
	    img = image.load_img(img, target_size=(224, 224))
	    img = image.img_to_array(img)
	    img = np.expand_dims(img, axis=0)
	    images.append(img)
	#stacking vertically for display
	images = np.vstack(images)
	classes = model.predict(images)
	print(classes)
	return(classes)

index = int(os.environ['LSB_JOBINDEX'])
print (index)
output_path='/hps/nobackup/birney/projects/ukb_eye_image_analysis/output_latest_model'
root_path='/hps/nobackup/sds/sds-ukb-geno/phenotypes/oct/png_store/'

count = 1
with open ('/hps/nobackup/sds/sds-ukb-geno/phenotypes/oct/21017/ukb44854_21017.bulk') as file:
	for line in file:
		if count == index:
			items = line.rstrip().split(" ")
			path_str = "0" + items[0]
			path_str = "/".join([path_str[i:i+2] for i in range(0, len(path_str), 2)])
			path_str = root_path + path_str + "/" + items[0] 
			image_folders = os.listdir(path_str)
			for folder in image_folders:
				fit = fit_model(path_str + "/" + folder)
				outputfile_name = output_path + "/" + items[0] + "_" + folder + ".txt"
				print (outputfile_name)
				np.savetxt(outputfile_name, fit, fmt = '%.6f')
		count = count+1