from utils import retrieve_data, plot_image
from model import predict
import keras
import numpy as np
import random
import matplotlib.pyplot as plt


def interactive_visual():
	X_train, Y_train, X_test, Y_test = retrieve_data()
	model = keras.models.load_model('Deep_NN_MNIST')
	#model = keras.models.load_model('Dropout_NN_mnist')
	evaluation = model.evaluate(X_test, Y_test)
	print('accuracy on test set: ' + str(evaluation[1]))
	plt.ion()
	while True:
		rand = random.randint(0, X_test.shape[0])
		prediction = int(predict(model, X_test[rand]))
		print('NN prediction: ' + str(prediction))
		plt.imshow(np.reshape(X_test[rand], (28, 28)))
		x = input('press enter to display new image or type exit: ')
		if x == 'exit':
			break
			plt.close()
		elif x == '':
			plt.draw()
		plt.clf()

if __name__ == '__main__':
	interactive_visual()
	
