from Layer import Layer
import numpy as np

class Activation(Layer):
	def __init__(self, act, dact):
		self.act = act
		self.dact = dact

	def forward(self, inp):
		self.input = inp
		return self.act(self.input)

	def backward(self, o_grad, l_rate):
		print("o", o_grad.shape)
		print("dact", self.dact(self.input)[np.newaxis, :].shape)
		return o_grad @ self.dact(self.input)[np.newaxis, :]

class ReLU(Activation):
	def __init__(self):
		def reLU(x):
			return np.maximum(x, 0)

		def dreLU(x):
			return x > 0
		
		super().__init__(reLU, dreLU)

class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			return 1 / (1 + np.exp(-x))

		def dsigmoid(x):
			sig = 1 / (1 + np.exp(-x))
			return sig * (1 - sig)
		
		super().__init__(sigmoid, dsigmoid)

class SoftMax(Activation):
	def __init__(self):
		def softmax(x):
			tmp = np.exp(x)
			self.output = tmp / np.sum(tmp)
			return self.output

		def dsoftmax(x):
			n = np.size(self.output)
			return np.dot((np.identity(n) - self.output.T) * self.output, o_grad)

		super().__init__(softmax, dsoftmax)


__all__ = ["ReLU", "Sigmoid", "SoftMax"]
