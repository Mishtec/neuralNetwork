import numpy as np

# neuralNetwork
class neuralNetwork:
    def __init__(self, sizes):
        np.random.seed(6795)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # 1 x 2, 1 x 1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] 
        # random, sizes[:-1] last column, sizes[1:] for hidden layer and ouptut layer
        # 2 x 2, 1 x 2
      
        #print(self.weights)
        #print(self.biases)

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases


    # Sigmoid function = Activation function 
    def sigmoid(self, x):
        return  1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    # Feed forward
    # Formula: 
    # Let aj denote the output of unit j and let wi,j be the weight attached to the link from unit i to unit j; then we have
    # aj = gj (Σi wi,j aj) == gj (inj)
    # Where gj is a nonlinear activation function (sigmoid function) associated with unit j and inj is the weighted sum of the inputs to unit j.

    def feedforward(self, inputs):
        for b, w in zip(self.biases, self.weights):
            inputs = self.sigmoid(np.dot(w, inputs ) + b)

        #  sigmoid func for the σ = weighted sums
        return inputs

    # cost function
    # [output - expected output]^2 
    def cost_function(self, output, expected_out):
        cost = (output - expected_out)**2
        return cost

    # avg cost of all training data = performance measure 
    def avg_cost(self, output, expected_out):
        return np.mean(self.cost_function(self, output, expected_out))


    # Compute gradient of Cost function
    # Small step in -gradient of Cost function direction
    # Repeat until we find the minimum



    # back propogation = for computing gradient efficiently 


    



    # Training dataset for bias and weights 
    # def train(self, ):







        











# weights, biases, and the activation function must all be implemented by you
# a) How many hidden layers have you used?
# b) What are the weights and biases of each node?    



  

       
  


