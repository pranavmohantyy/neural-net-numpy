import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        return X @ self.W + self.b

    def compute_gradients(self, X, y_true, y_pred):
        dW = X.T @ (y_pred - y_true) / y_true.shape[0]
        db = np.sum(y_pred - y_true, axis=0, keepdims=True) / y_true.shape[0]
        dX = (y_pred - y_true) @ self.W.T
        return dW, db, dX