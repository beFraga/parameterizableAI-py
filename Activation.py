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


__all__ = ["ReLU", "Sigmoid"]
