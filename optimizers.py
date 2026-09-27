import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update_params(self, layer):
        for name, param in layer.__dict__.items():
            if isinstance(param, np.ndarray):
                if name not in self.velocities:
                    self.velocities[name] = np.zeros_like(param)
                self.velocities[name] = self.momentum * self.velocities[name] - self.learning_rate * param
                layer.__dict__[name] += self.velocities[name]