import numpy as np

class WTANetwork:
    def __init__(self, num_neurons, learning_rate):
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_neurons, 2)  # Инициализация весов случайными значениями

    def fit(self, X, y):
        for x, target in zip(X, y):
            # Вычисляем активацию нейронов
            activations = np.dot(self.weights, x)
            # Находим индекс нейрона с максимальной активацией
            winner_index = np.argmax(activations)
            # Обновляем веса только для победившего нейрона
            self.weights[winner_index] += self.learning_rate * (x - self.weights[winner_index])

    def predict(self, X):
        predictions = []
        for x in X:
            activations = np.dot(self.weights, x)
            winner_index = np.argmax(activations)
            predictions.append(winner_index)
        return np.array(predictions)