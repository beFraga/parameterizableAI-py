from Layer import Layer

class Losses(Layer):
	def __init__(self, loss, dloss):
		self.loss = loss
		self.dloss = dloss

	def forward(self, true, pred):
		return self.loss(true, pred)
	
	def backward(self, true, pred):
		return self.dloss(true, pred)

class MSE(Losses):
	def __init__(self):
		def mse(true, pred):
			return np.mean(np.power(true - pred, 2))
		
		def dmse(true, pred):
			return 2 * (true - pred) / np.size(true)

		super().__init__(self, mse, dmse)


__all__ = ["MSE"]
