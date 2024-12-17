from Layer import Layer

class Activation(Layer):
	def __init__(self, act, dact):
		self.act = act
		self.dact = dact

	def forward(self, inp):
		self.input = inp
		return self.act(self.input)

	def backward(self, o_grad, l_rate):
		return o_gradient * self.dact

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
			exps = np.exp(x - exp.max(x, axis=-1, keepdims=True))
			return exps / np.sum(exps, axis=-1, keepdims=True)

		def dsoftmax(x):
			return np.diagflat(x) - np.dot(x, x.T)

		super().__init__(softmax, dsoftmax)


__all__ = ["ReLU", "Sigmoid", "SoftMax"]
