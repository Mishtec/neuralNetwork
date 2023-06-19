import neuralNetwork
import numpy as np


# Usage
neuralNetwork = neuralNetwork.neuralNetwork(sizes = [2, 2, 1])
# 2 neurons in inputs layer(s)
# 2 hidden layer(s)
# 1 output layer(S)

dataset = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
   ])

X = dataset[:, :-1] # all except last
y = dataset[:, -1] # last column

print(dataset)
print(X)
print (y)

learning_rate = 0.1
epochs = 10
# total number of iterations of all the training data in one cycle for training

# neuralNetwork.train(X, y, epochs, learning_rate)

# Test 
expected_output = neuralNetwork.feedforward(X.T)

# Print Results
for x, y in zip(X, expected_output.T):
    print(f"Input: {x}  Output: {y}")






