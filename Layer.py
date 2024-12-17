from scipy import signal

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
		w_grad = np.dot(o_grad, self.input.T)
		i_grad = np.dot(self.weights.T, o_grad)
		self.weights -= w_grad * l_rate
		self.bias -= o_grad * l_rate
		return i_grad

class Convolutional(Layer):
	def __init__(self, kernel, bias):
		self.kernel = kernel
		self.bias = bias

	def forward(inp):
		self.input = inp
		self.output = np.copy(self.bias)
		self.output += signal.correlate2d(self.input, self.kernel, "valid")
		return self.output

	def backward(o_grad, l_rate):
		k_grad = np.zeros(self.kernel.shape)
		i_grad = np.zeros(self.input.shape)
		k_grad = signal.correlate2d(self.input, o_grad, "valid")
		i_grad += signal.correlate2d(o_grad, self.kernel, "full")
		self.kernel -= k_grad * l_rate
		self.bias -= o_grad * l_rate
		return i_grad 
