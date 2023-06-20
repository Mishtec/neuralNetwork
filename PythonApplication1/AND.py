import math

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Neural network class
class AndGateNN:
    def __init__(self):
        # Define weights and biases
        self.hidden_weights = [20, 20]
        self.hidden_bias = -30
        self.output_weight = [20]
        self.output_bias = -10

    # Forward propagation
    def forward(self, x1, x2):
        # Hidden layer computation
        hidden_output = sigmoid(self.hidden_weights[0] * x1 + self.hidden_weights[1] * x2 + self.hidden_bias)

        # Output layer computation
        output = sigmoid(self.output_weight[0] * hidden_output + self.output_bias)

        return output

# Testing the AND gate neural network
and_gate_nn = AndGateNN()

# Test inputs
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Perform AND gate operation
for x1, x2 in inputs:
    output = and_gate_nn.forward(x1, x2)
    print(f"Input: ({x1}, {x2}) - Output: {output:.2f}")
