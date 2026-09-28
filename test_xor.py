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
for epoch in range(1000):
    y_pred = nn.forward(X)
    loss = nn.compute_loss(y_true, y_pred)
    gradients = nn.backward(X, y_true, y_pred)
    nn.optimizer.update_params(nn.layers[0])
    nn.optimizer.update_params(nn.layers[1])

# Testing
predictions = nn.forward(X)
print(predictions)
