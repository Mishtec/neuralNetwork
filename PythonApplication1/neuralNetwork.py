import numpy as np

# neuralNetwork
class neuralNetwork:
    def __init__(self, sizes):
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # 1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # random, sizes[:-1] last column, sizes[1:] for hidden layer and ouptut layer
        # print(self.weights)
        # print(self.biases)


    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_biases(self, biases):
        self.biases = biases

    def get_biases(self):
        return self.biases


    # Sigmoid function = Activation function 
    def sigmoid(x):
        return  1 / (1 + np.exp(-x))

    # Feed forward
    # Formula: 
    # Let aj denote the output of unit j and let wi,j be the weight attached to the link from unit i to unit j; then we have
    # aj = gj (Σi wi,j aj) == gj (inj)
    # Where gj is a nonlinear activation function (sigmoid function) associated with unit j and inj is the weighted sum of the inputs to unit j.

    def feedforward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        #  sigmoid func for the σ = weighted sums
        return sigmoid(weighted_sum)

    # Training dataset for bias and weights 
    #def train():







        











# weights, biases, and the activation function must all be implemented by you
# a) How many hidden layers have you used?
# b) What are the weights and biases of each node?    



  

       
  


