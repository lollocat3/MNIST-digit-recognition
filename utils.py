from tensorflow.keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt

def retrieve_data(): 
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = np.reshape(X_train, (X_train.shape[0], 784))
	X_test = np.reshape(X_test, (X_test.shape[0], 784))
	Y_train = keras.utils.to_categorical(Y_train, 10)
	Y_test = keras.utils.to_categorical(Y_test, 10)
	return X_train, Y_train, X_test, Y_test

def plot_image(array, show = True):  #array is a two-dimensional array of pixel values
	if array.ndim == 1:
		array = np.reshape(array, (28, 28))
	plt.imshow(array, cmap = 'gray')
	if show:
		plt.show()
	return

def display_data(X):
	f, axarr = plt.subplots(10, 10)
	plt.subplots_adjust(wspace = 0, hspace = 0)
	for i in range(10):
		for j in range(10):
			axarr[i, j].axis('off')
			axarr[i, j].imshow(X[np.random.randint(0, X.shape[0])])
	plt.show()
	return 


