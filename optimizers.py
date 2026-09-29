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
                self.velocities[name] = self.momentum * self.velocities[name] - self.learning_rate * layer.__dict__[name]
                layer.__dict__[name] += self.velocities[name]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update_params(self, layer):
        self.t += 1
        for name, param in layer.__dict__.items():
            if isinstance(param, np.ndarray):
                if name not in self.m:
                    self.m[name] = np.zeros_like(param)
                    self.v[name] = np.zeros_like(param)
                grad = layer.__dict__[name + '_grad']
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                layer.__dict__[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
