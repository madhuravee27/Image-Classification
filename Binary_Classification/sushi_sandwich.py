import cv2
import numpy as np
import glob
import tensorflow as tf

import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

def load_image():
	#train_file_location = input("Enter the sushi images file location")
	#valid_file_location = input("Enter the sushi images file location")

	train_file_location = "C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\SushiSandwich\\sushisandwich\\train"
	valid_file_location = "C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\SushiSandwich\\sushisandwich\\test"

	train_images_sushi = glob.glob(train_file_location + '\\sushi\\*.jpg')
	train_images_sandwich = glob.glob(train_file_location + '\\sandwich\\*.jpg')
	train_images = train_images_sushi + train_images_sandwich
	train_X = []
	#i = 0
	for image in train_images:
		image = tf.image.resize_image_with_crop_or_pad(image = cv2.imread(image), target_height = 244, target_width = 244).eval()
		train_X.append(image)
		#i += 1
		#if (i == 50):
			#break
	train_X = np.stack(train_X).reshape((-1,244,244,3))
	train_Y = np.concatenate((np.zeros(len(train_images_sushi)), np.ones(len(train_images_sandwich))), axis = None).reshape((-1,1))
	#train_Y = np.zeros(train_X.shape[0])
	shuffle = np.random.permutation(train_X.shape[0])
	train_X = train_X[shuffle]
	train_Y = train_Y[shuffle]

	valid_images_sushi =  glob.glob(train_file_location + '\\sushi\\*.jpg')
	valid_images_sandwich = glob.glob(train_file_location + '\\sandwich\\*.jpg')
	valid_images = valid_images_sushi + valid_images_sandwich
	valid_X = []
	#i = 0
	for image in valid_images:
		image = tf.image.resize_image_with_crop_or_pad(image = cv2.imread(image), target_height = 244, target_width = 244).eval()
		valid_X.append(image)
		#i += 1
		#if (i == 50):
			#break
	valid_X = np.stack(valid_X).reshape((-1,244,244,3))
	valid_Y = np.concatenate((np.zeros(len(valid_images_sushi)), np.ones(len(valid_images_sandwich))), axis = None).reshape((-1,1))
	#valid_Y = np.zeros(valid_X.shape[0])
	shuffle = np.random.permutation(train_X.shape[0])
	valid_X = valid_X[shuffle]
	valid_Y = valid_Y[shuffle]

	return train_X, train_Y, valid_X, valid_Y

def model_build(input_shape):
	X_input = Input(input_shape)
	X = Conv2D(filters = 96, kernel_size = (11,11), strides = (4,4), activation = 'relu')(X_input)
	X = MaxPooling2D()(X)
	X = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), activation = 'relu')(X)
	X = MaxPooling2D()(X)
	X = Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = 'relu')(X)
	X = MaxPooling2D()(X)
	X = Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same')(X)
	X = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same')(X)
	'''
	shape = X.shape.as_list()
	levels = [6, 3, 2, 1]
	outputs = []
	for l in levels:
		out = MaxPooling2D(pool_size = (np.ceil(int(shape[1])/l),np.ceil(int(shape[2])/l)), 
			strides = (np.floor(int(shape[1])/l),np.floor(int(shape[2])/l)), padding = 'same')(X)
		out = Flatten()(out)
		outputs.append(out)
	X = np.asarray(outputs)
	'''
	X = MaxPooling2D()(X)
	X = Flatten()(X)
	X = Dense(4096)(X)
	X = Dense(4096)(X)
	X = Dense(1, activation = 'sigmoid')(X)
	model = Model(inputs = X_input, outputs = X, name='AlexNet')

	return model

def main():
	with tf.Session() as sess:
		train_X, train_Y, test_X, test_Y = load_image()
		alexNet = model_build(train_X.shape[1:])
		alexNet.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
		epochs = 10;
		batch_size = 128;
		alexNet.fit(train_X, train_Y, epochs= epochs, batch_size = batch_size)
		prediction = alexNet.evaluate(test_X, test_Y, batch_size = 32, verbose = 1, sample_weight=None)
		print("###############\nTest set\n###############")
		print ("Test Loss = " + str(prediction[0]))
		print ("Test Accuracy = " + str(prediction[1]))	

if __name__== "__main__":
  main()
