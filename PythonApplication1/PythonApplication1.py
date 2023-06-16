import Model
import Controller
import View
import neuralNetwork
import numpy as np


# Usage
model = Model.Model()
view = View.View()
controller = Controller.Controller(model, view)
neuralNetwork = neuralNetwork.neuralNetwork(sizes = [2, 2, 1])
# 2 inputs layer(s)
# 2 hidden layer(s)
# 1 output layer(S)

dataset = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
   ])

# Update the data through the controller
controller.update_dataset(dataset)
controller.update_biases_wieghts(neuralNetwork.get_weights(), neuralNetwork.get_biases())






