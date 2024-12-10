class Layer:
	def __init__(self):
		self.input = None
		self.output = None

	def forward():
		pass # nao precisa fazer nada
	
	def backward():
		pass # nao precisa fazer nada

class Dense(Layer):
	def __init__(self, weights, bias):
		self.weights = weights
		self.bias = bias

	def forward(inp):
		self.input = inp
		return self.weights * self.input + self.bias

	def backward(o_grad, l_rate):
		#w_grad = 
		#inp_grad = 
		self.weights -= w_grad * l_rate
		self.bias -= o_grad * l_rate


