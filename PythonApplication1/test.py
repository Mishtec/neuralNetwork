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

# 2 neurons in input layer
# 2 neurons in hidden layer -> 2 layers
# 1 neurons in output layer
layer1 = neuralNetwork.neuralNetwork(X.shape[1], 2)
layer2 = neuralNetwork.neuralNetwork(2, 2)
layer3 = neuralNetwork.neuralNetwork(2, 1)


print("weights 1:\n", layer1.get_weights()) # these are only neuron weights 
print("biases 1:\n",layer1.get_biases())
output1 = layer1.feedforward(X)
print("output1:\n", output1)


print("weights 2:\n", layer2.get_weights())
output2 = layer2.feedforward(output1)
print("output2:\n", output2)


print("weights 3:\n", layer3.get_weights()) 
print("biases 3:\n",layer3.get_biases())
output3 = layer3.feedforward(output2)
print("output3:\n", output3)


cost1 = layer1.cost_function(output1, y.T)
print("cost 1:\n", cost1)
cost2 = layer2.cost_function(output2, y.T)
print("cost 2:\n", cost2)
cost3 = layer3.cost_function(output3, y.T)
print("cost 3:\n", cost3)

avg_cost3 = layer3.avg_cost(output3, y.T)
print("avg_cost 3:\n", avg_cost3)


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

print("weights 3:\n", layer3.get_weights()) 
print("biases 3:\n",layer3.get_biases())
print("output3:\n", output3.round())

cost3 = layer3.cost_function(output3, y.T)
print("cost 3:\n", cost3)

avg_cost3 = layer3.avg_cost(output3, y.T)
print("avg_cost 3:\n", avg_cost3)


 # create a more user friendly appoach 


