# Model
class Model:
    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.weights = None
        self.biases = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    def set_inputs(self):
        self.X = self.dataset[:, :-1] # all except last

    def get_inputs(self):
        return self.X
  
    def set_outputs(self):
        self.y = self.dataset[:, -1] # last column

    def get_outputs(self):
        return self.y

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_biases(self, biases):
        self.biases = biases

    def get_biases(self):
        return self.biases




