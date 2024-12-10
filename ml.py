import pandas as pd
import numpy as np
import scipy.stats as sps
from Activation import *
from Losses import *

class AI:
	def __init__(self, new: bool, loss=MSE, act=ReLU, config=None, mH=3, nH=10, l_rate=0.1, epochs=100):
		self.act = None
		self.loss = None
		self.network = []
		self.epochs = None
		self.l_rate = None
		self.set_params(mH, nH, act, loss, epochs, l_rate) if new else self.init_params(config)

	def init_params(self, config):
		with open('config.json') as fp:
			listJson = json.load(fp)
		{act, loss, epochs, l_rate, mH, nH} = listJson[config]
		self.act = classes[act]
		self.loss = classes[loss]
		self.epochs = int(epochs)
		self.l_rate = float(l_rate)

		# -------- FALTA PEGAR O NH E MH E FAZER A SELF.NETWORK --------------------
		

	def set_params(self, mH, nH, act, loss, epochs, l_rate):
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
		listJson.append(c)

		with open('config.json', 'w') as json_file:
			json.dump(listJson, json_file, indent=4, separators=(',', ': '))


	def get_params(self):
		with open('config.json') as fp:
			jsonObj = json.load(fp)
		for k,v in jsonObj.entries():
			print(f"{k}:{v}")

	def train(self, xtrain, ytrain):
		for e in range(self.epochs)
			error = 0
				for x,y in zip(xtrain, ytrain):
					output = predict(network, x)
					error += self.loss.forward(y, output)

					grad = self.loss.backward(y, output)
					for layer in reversed(newtwork):
						grad = layer.backward(grad, self.l_rate)

			error /= len(xtrain)
			print(f"{e + 1}/{self.epochs}, error = {error}"
	
	def predict(self, network, inp):
		output = inp
		for layer in network:
			output = layer.forward(output)
		
		return output


classes = {"Sigmoid": Sigmoid, "ReLU": ReLU, "MSE": MSE}



def main():
	print('Rodando')


if __name__ == "__main__":
	main()
