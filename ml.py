import pandas as pd
import numpy as np
import json
import scipy.stats as sps
from Activation import *
from Losses import *

class AI:
	def __init__(self, config: int, new: bool, loss=MSE, act=ReLU, mH=3, nH=10, l_rate=0.1, epochs=100):
		self.act = None
		self.loss = None
		self.network = []
		self.epochs = None
		self.l_rate = None
		self.set_params(config, mH, nH, act, loss, epochs, l_rate) if new else self.init_params(config)

	def init_params(self, n):
		with open('config.json') as fp:
			listJson = json.load(fp)
		if not listJson[n]:
			return
		obj = listJson[n]
		self.act = classes[obj["act"]]
		self.loss = classes[obj["loss"]]
		self.epochs = int(obj["epochs"])
		self.l_rate = float(obj["l_rate"])

		# -------- FALTA PEGAR O NH E MH E FAZER A SELF.NETWORK --------------------
		

	def set_params(self, n, mH, nH, act, loss, epochs, l_rate):
		c = {
			mH: mH,
			nH: nH,
			act: act,
			loss: loss,
			epochs: epochs,
			l_rate: l_rate
		}

		with open('config.json') as fp:
			listJson = json.load(fp)
		if listJson[n]:
			return

		listJson[n] = c

		with open('config.json', 'w') as json_file:
			json.dump(listJson, json_file, indent=4, separators=(',', ': '))
		self.init_params(n)

	def remove_param(self, n):
		with open('config.json') as fp:
			listJson = json.load(fp):

		listJson[n] = {}

		with open('config.json', 'w') as json_file:
			json.dump(listJson, json_file, indent=4, separators(',', ': '))
		

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


classes = {"Sigmoid": Sigmoid, "ReLU": ReLU, "MSE": MSE}



def main():
	print('Rodando')
	AI.get_params()


if __name__ == "__main__":
	main()
