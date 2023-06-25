import neuralNetwork
import numpy as np


dataset = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
   ])

X = dataset[:, :-1] # all except last
#y = dataset[:, -1] # last column, could not transpose
y = np.array([[0,0,0,1]])

# 2 neurons in input layer 0 
# 2 neurons in hidden layer 1 
# 1 neuron in hidden layer 2
# 1 neurons in output layer 3 


layer1 = neuralNetwork.neuralNetwork(X.shape[1], 2)
layer2 = neuralNetwork.neuralNetwork(2, 1)
layer3 = neuralNetwork.neuralNetwork(1, 1)


def print_weights_biases(layer, output, n):
    print("weights ", n, " :\n", layer.get_weights(), "\n") # these are only neuron weights 
    print("biases ", n, " :\n", layer.get_biases(), "\n")
    print("output ", n, " :\n", output, "\n", "\n")


output1 = layer1.feedforward(X)
print_weights_biases(layer1, output1, 1)


output2 = layer2.feedforward(output1)
print_weights_biases(layer2, output2, 2)


output3 = layer3.feedforward(output2)
print_weights_biases(layer3, output2, 3)


cost1 = layer1.cost_function(output1, y.T)
print("cost 1:\n", cost1, "\n")
cost2 = layer2.cost_function(output2, y.T)
print("cost 2:\n", cost2, "\n")
cost3 = layer3.cost_function(output3, y.T)
print("cost 3:\n", cost3, "\n")

avg_cost3 = layer3.avg_cost(output3, y.T)
print("avg_cost 3:\n", avg_cost3, "\n")


learning_rate = 0.1

# total number of iterations of all the training data in one cycle for training
epochs = 10000

for epoch in range(epochs):
    d_weights3, d_bias3 = layer3.back_propogation(output2, output3, layer3.cost_function(output3, y.T), learning_rate)
    d_weights2, d_bias2 = layer2.back_propogation(output1, output2, layer2.cost_function(output2, y.T), learning_rate)
    d_weights1, d_bias1 = layer1.back_propogation(X, output1, layer1.cost_function(output1, y.T), learning_rate)
   # print("d_weights 3:\n", d_weights3)
   # print("d_weights 2:\n", d_weights2) 
   # print("d_weights 1:\n", d_weights1) 
   # print("d_bias 3:\n", d_bias3)
   # print("d_bias 2:\n", d_bias2)
   # print("d_bias 1:\n", d_bias1)
    output1 = layer1.feedforward(X)
    output2 = layer2.feedforward(output1)
    output3 = layer3.feedforward(output2)


print_weights_biases(layer3, output3, 3)
print("output3:\n", output3.round(), "\n")

cost3 = layer3.cost_function(output3, y.T)
print("cost 3:\n", cost3, "\n")

avg_cost3 = layer3.avg_cost(output3, y.T)
print("avg_cost 3:\n", avg_cost3, "\n")




