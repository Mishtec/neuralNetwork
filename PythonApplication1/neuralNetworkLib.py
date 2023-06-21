from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
# Dense layers for hidden layers and output node
# Sequential type model
#  


import numpy as np

x_train = np.array([[1, 1, 1, 0, 0],
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

y_train = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Use the following testing set to check how good the neural net is in classification:

x_test = np.array([[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1],
                     [1, 1, 0, 1, 0],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]])

y_test = np.array([1, 1, 0, 0, 1, 1, 1, 1])

# 1st define the type of the model
model = Sequential()

# add input layer 
model.add(Input(shape = x_train[0].shape))

# add hidden layer(s)
# h1. Nodes = 10, 
# activation options: relu: rectified linear, sigmoid
# rectified linear: 
model.add(Dense(10, activation = "relu"))

# h2. Nodes = 10 
model.add(Dense(10, activation = "relu"))

# output layer 
# activation = sigmiod -> 0 or 1
model.add(Dense(1, activation = "sigmoid"))

model.summary()

# use compile 
# optimizer = "adam" algorithim where it uses gradient descent 
# cost / loss function = binary_crossentropy; calculates loss between true values and predicted 
# metrics = frequency in which y_true = y_predict -> total outptut / n_count of predictions  

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = "binary_accuracy")

# 1000 iterations

model.fit(x_train, y_train, epochs = 1000)


x_test_output = model.predict(x_test)

print(x_test_output)

print(x_test_output.round())


# any neuon with a non-linear activation function 
# can fit a non-linear problem
# relu vs sigmoid 
