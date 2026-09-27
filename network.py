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

    def backward(self, X, y_true, y_pred):
        for layer in reversed(self.layers):
            dW, db, dX = layer.compute_gradients(X, y_true, y_pred)
            self.optimizer.update_params(layer)
            X = dX

    def train(self, X, y, epochs=10, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                self.backward(X_batch, y_batch, y_pred)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")