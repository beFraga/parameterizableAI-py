import pandas as pd
import numpy as np
import scipy.stats as sps

class AI:
	def __init__(self, new):
		self.I
		self.W
		self.O
		self.act
		self.deriv_act
		self.init_params() if new else self.set_params()

	def act_function(self, X):
		return self.act(X)

	def deriv_act_function(self, X):
		return self.deriv_act(X)

	def init_params():
		pass

	def set_params():
		pass

class Layer:
	def __init__(self):
		self.input = None
		self.output = None

	def forward():
		pass # nao precisa fazer nada
	
	def backward():
		pass # nao precisa fazer nada

class Activation(Layer):
	def __init__(self, act, dact):
		self.act = act
		self.dact = dact

	def forward(self, inp):
		self.input = inp
		return self.act(self.input)

	def backward(self, output_gradient, alpha):
		return np.multiply(output_gradient, self.dact)

class ReLU(Activation):
	def __init(self):
		def reLU(x):
			return np.maximum(x, 0)

		def dreLU(x):
			return x > 0
		
		super().__init__(self, reLU, dreLU)

class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			pass

		def dsigmoid(x):
			pass
		
		super().__init__(self, sigmoid, dsigmoid)

def feed_forward():



def back_propagation():



def run():
