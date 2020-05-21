from utils import retrieve_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt


def model(summary = False):
	model = Sequential()
	model.add(Dense(100, input_dim=784, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5)))
	model.add(Dense(32, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5)))
	model.add(Dense(10, activation = 'softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	if summary:
		model.summary()
	return model

def model_dropout(summary = False):
	model = Sequential()
	model.add(Dense(100, input_dim=784, activation='sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5)))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation = 'softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	if summary:
		model.summary()
	return model

def train(model, X_train, Y_train, epochs = 2, batch_size = 10, save = False, 
	name = 'NN_mnist', learning_curve = False):
	history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, verbose = 1)
	if save:
		model.save(name)
	if learning_curve:
		fig, ax = plt.subplots(2)
		ax[0].plot([i for i in range(epochs)], history.history['accuracy'])
		ax[1].plot([i for i in range(epochs)], history.history['loss'])
		plt.show()
	return history

def predict(model, X):
	import numpy as np
	if X.ndim == 1:
		X = np.expand_dims(X, axis = 0)
	return model.predict_classes(X)





