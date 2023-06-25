import numpy as np

# neuralNetwork
class neuralNetwork:
    def __init__(self, n_inputs, n_neurons):
        np.random.seed(4)
        self.biases = np.zeros((1, n_neurons))
        self.weights = np.random.randn(n_inputs, n_neurons)
        
     
    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    # Sigmoid function = Activation function 
    # input x = weighted sum
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
        inputs = (self.sigmoid(np.dot(inputs, self.weights) + self.biases))
        #  sigmoid func for the σ = weighted sums = inputs
        return inputs

    # cost function
    # [output - expected output]^2 
    def cost_function(self, actual_output, expected_out):
        cost = (expected_out - actual_output)
        return cost

    # avg cost of all training data = performance measure 
    def avg_cost(self, output, expected_out):
        return np.mean(self.cost_function(output, expected_out)**2)


    # Compute gradient of Cost function
    # Small step in -gradient of Cost function direction
    # Repeat until we find the minimum
    # back propogation = for computing gradient efficiently 
    def back_propogation(self, inputs, weighted_sums, cost, learning_rate):
        d_weights = np.dot(inputs.T, cost * self.sigmoid_derivative(weighted_sums))
        d_bias = np.sum(cost * self.sigmoid_derivative(weighted_sums), axis =0)
         # Update the weights and bias
        self.weights += learning_rate * d_weights
        self.biases += learning_rate * d_bias
        return d_weights, d_bias
    








        











 



  

       
  


