import pandas as pd
import numpy as np
import json
import scipy.stats as sps
from PIL import Image as pili
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from Activation import *
from Losses import *
from Layer import Dense, Convolutional, Resize


'''
INTERFACE GRAFICA + PAINT
SALVAR IMAGEM
DADOS PARA TREINO
'''

class AI:
	def __init__(self, config: int, new: bool, loss="MSE", act="ReLU", mH=3, nH=10, l_rate=0.1, epochs=100, kernel=3):
		self.act = None
		self.loss = None
		self.network = []
		self.epochs = None
		self.l_rate = None
		self.set_params(config, mH, nH, kernel, act, loss, epochs, l_rate) if new else self.init_params(config)

	def init_params(self, n):
		with open('config.json') as fp:
			listJson = json.load(fp)
		if listJson[n] is None:
			return
		obj = listJson[n]
		self.act = classes[obj["act"]]
		self.loss = classes[obj["loss"]]
		self.epochs = int(obj["epochs"])
		self.l_rate = float(obj["l_rate"])
		bias = np.array(obj["bias"])
		weight = np.array(obj["weight"])
		biasC = np.array(obj["biasC"])
		kernel = np.array(obj["kernel"])
		wih = np.array(obj["wih"])
		who = np.array(obj["who"])
		biasO = np.array(obj["biasO"])
		

		network = []

		network.append(Convolutional(kernel, biasC))
		network.append(self.act)

		network.append(Resize(biasC.shape, np.prod(biasC.shape)))

		for i,b in enumerate(bias):
			if i == 0:
				network.append(Dense(wih, b))
			else:
				network.append(Dense(weight[i-1], b))
				network.append(self.act)

		network.append(Dense(who, biasO))
		network.append(self.act)
	
		print(network)
		self.network = network		
		
	def set_params(self, n, mH, nH, ker_s, act, loss, epochs, l_rate):
		
		if n >= 5:
			raise ValueError("O indice deve estar entre 0 e 4")
			
		bias = np.random.randn(mH+1, nH)
		weight = np.random.randn(mH, nH, nH)
		biasC = np.random.randn(28 - ker_s + 1, 28 - ker_s + 1)
		kernel = np.random.randn(ker_s, ker_s)
		wih = np.random.randn(nH, (28 - ker_s + 1)**2)
		who = np.random.randn(2, nH)
		biasO = np.random.randn(2)
		
		c = {
			"nH": nH,
			"mH": mH,
			"act": act,
			"loss": loss,
			"epochs": epochs,
			"l_rate": l_rate,
			"bias": bias.tolist(),
			"weight": weight.tolist(),
			"biasC": biasC.tolist(),
			"kernel": kernel.tolist(),
			"wih": wih.tolist(),
			"who": who.tolist(),
			"biasO": biasO.tolist()
		}
		try:
			with open('config.json') as fp:
				listJson = json.load(fp)
		except(FileNotFoundError, json.JSONDecodeError):
			listJson = [None] * 5
			print("Erro")


		if listJson[n] is not None:
			return

		listJson[n] = c

		with open('config.json', 'w') as json_file:
			json.dump(listJson, json_file, indent=4, separators=(',', ': '))

		self.init_params(n)

	@staticmethod
	def remove_param(n):
		with open('config.json') as fp:
			listJson = json.load(fp)

		listJson[n] = None

		with open('config.json', 'w') as json_file:
			json.dump(listJson, json_file, indent=4, separators=(',', ': '))
		

	@staticmethod
	def get_params():
		with open('config.json') as fp:
		 	listJson = json.load(fp)
		for i in listJson:
			print(i)

	def train(self, xtrain, ytrain):
		for e in range(self.epochs):
			error = 0
			for x,y in zip(xtrain, ytrain):
				x = x.reshape(28,28)
				output = self.predict(x)
				output = output[:, np.newaxis]
				error += self.loss.forward(y, output)
				print("y",y, y.shape)
				print("o",output, output.shape)
				print("e",error)
				grad = self.loss.backward(y, output)
				print("grad", grad.shape)
				for layer in reversed(self.network):
					print(layer)
					grad = layer.backward(grad, self.l_rate)
					print(grad)

			error /= len(xtrain)
			print(f"{e + 1}/{self.epochs}, error = {error}")
	
	def predict(self, inp):
		output = inp
		for layer in self.network:
			output = layer.forward(output)
		
		return output

	def test(xtest, ytest):
		for x,y in zip(xtest, ytest):
			output = self.predict(x)
			print(f"{np.argmax(output)} / {np.argmax(y)}")

	def image_to_np(self):
		img = pili.open('temp.png')
		imgr = img.resize((50,50))
		imgnp = np.asarray(img)
		imgnp = np.dot(imgnp[...,:3], [0.2989, 0.587, 0.114])
		return imgnp

	def process(x, y, limit):
		zero_index = np.where(y == 0)[0][:limit]
		one_index = np.where(y == 1)[0][:limit]
		all_indices = np.hstack((zero_index, one_index))
		all_indices = np.random.permutation(all_indices)
		x, y = x[all_indices], y[all_indices]
		x = x.reshape(len(x), 1, 28, 28)
		x = x.astype("float32") / 255
		y =	to_categorical(y)
		y = y.reshape(len(y), 2, 1)
		return x, y


classes = {"Sigmoid": Sigmoid(), "ReLU": ReLU(), "MSE": MSE()}



def main():
	print('Rodando')
	AI.remove_param(0)
	ai = AI(0, True)
	(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
	xtrain, ytrain = AI.process(xtrain, ytrain, 100)
	xtest, ytest = AI.process(xtest, ytest, 100)
	ai.train(xtrain, ytrain)
	ai.test(xtest, ytest)	
	


if __name__ == "__main__":
	main()
