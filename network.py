import numpy as np
from layers import DenseLayer
from losses import mean_squared_error, mean_squared_error_derivative
from optimizers import SGD

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = SGD()

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_true, y_pred):
        self.loss = mean_squared_error(y_true, y_pred)
        return self.loss

    def backward(self, X, y_true):
        loss_derivative = mean_squared_error_derivative(y_true, self.forward(X))
        for layer in reversed(self.layers):
            loss_derivative = layer.compute_gradients(X, y_true, loss_derivative)
            self.optimizer.update_params(layer)

    def train(self, X, y_true):
        y_pred = self.forward(X)
        loss = self.compute_loss(y_true, y_pred)
        self.backward(X, y_true)
        return loss