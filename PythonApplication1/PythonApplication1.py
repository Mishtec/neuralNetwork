import Model
import Controller
import View
import neuralNetwork
import numpy as np


# Usage
model = Model.Model()
view = View.View()
controller = Controller.Controller(model, view)
neuralNetwork = neuralNetwork.neuralNetwork()


dataset = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
   ])

# Update the data through the controller
controller.update_dataset(dataset)




