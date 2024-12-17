import pandas as pd
import numpy as np
import json
import scipy.stats as sps
from PIL import Image as pili
from Activation import *
from Losses import *
from Layer import Dense, Convolutional


'''
INTERFACE GRAFICA + PAINT
SALVAR IMAGEM
DADOS PARA TREINO
'''

class AI:
	def __init__(self, config: int, new: bool, loss="MSE", act="ReLU", mH=3, nH=10, l_rate=0.1, epochs=100, kernel=5):
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

		network = []

		network.append(Convolutional(kernel, biasC))
		network.append(self.act)

		for i,v in enumerate(weight):
			if i == len(weight):
				network.append(Dense(v, bias[i]))
				network.append(SoftMax())
			else:
				network.append(Dense(v, bias[i]))
				network.append(self.act)
		print(network)
		self.network = network		
		

	def set_params(self, n, mH, nH, ker_s, act, loss, epochs, l_rate):
		
		if n >= 5:
			raise ValueError("O indice deve estar entre 0 e 4")
			
		bias = np.random.randn(nH, mH)
		weight = np.random.randn(mH, nH, nH)
		biasC = np.random.randn(200 - ker_s + 1, 200 - ker_s + 1)
		kernel = np.random.randn(ker_s, ker_s)
		#200 é o tamanho do input
		
		c = {
			"act": act,
			"loss": loss,
			"epochs": epochs,
			"l_rate": l_rate,
			"bias": bias.tolist(),
			"weight": weight.tolist(),
			"biasC": biasC.tolist(),
			"kernel": kernel.tolist()
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
				output = predict(x)
				error += self.loss.forward(y, output)

				grad = self.loss.backward(y, output)
				for layer in reversed(self.network):
					grad = layer.backward(grad, self.l_rate)

			error /= len(xtrain)
			print(f"{e + 1}/{self.epochs}, error = {error}")
	
	def predict(self, inp):
		output = inp
		for layer in self.network:
			output = layer.forward(output)
		
		return output

	def image_to_np(self):
		img = pili.open('temp.png')
		imgnp = np.asarray(img)
		imgnp = np.dot(imgnp[...,:3], [0.2989, 0.587, 0.114])
		return imgnp


classes = {"Sigmoid": Sigmoid(), "ReLU": ReLU(), "MSE": MSE()}



def main():
	print('Rodando')
	AI.remove_param(0)
	ai = AI(0, True)
	


if __name__ == "__main__":
	main()
