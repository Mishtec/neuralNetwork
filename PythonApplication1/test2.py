import tensorflow as tf
from tensorflow import keras
import numpy as np

dataset = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0],
                    [1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0],
                    [1, 0, 1, 1, 1]])

y = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Use the following testing set to check how good the neural net is in classification:

test_data = np.array([[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1],
                     [1, 1, 0, 1, 0],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]])

test_outputs = np.array([1, 1, 0, 0, 1, 1, 1, 1])



