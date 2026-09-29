import numpy as np
from network import NeuralNetwork
from layers import DenseLayer

# Data for XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [1], [1], [0]])

# Build the network
nn = NeuralNetwork()
nn.add_layer(DenseLayer(2, 4))
nn.add_layer(DenseLayer(4, 1))

# Training
nn.train(X, y_true, epochs=1000)

# Testing
predictions = nn.forward(X)
print(predictions)