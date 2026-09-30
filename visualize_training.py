import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def add_data(self, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def plot(self):
        epochs = range(1, len(self.losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.losses, 'b-', label='Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accuracies, 'g-', label='Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()