import numpy as np

class KohonenNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.3):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(output_size, input_size)

    def train(self, data, epochs):
        for epoch in range(epochs):
            for sample in data:
                # Находим ближайший нейрон (по Евклидовой или Манхэттенской мере)
                distances = np.linalg.norm(self.weights - sample, axis=1)
                winner_index = np.argmin(distances)

                # Обновление весов
                self.weights[winner_index] += self.learning_rate * (sample - self.weights[winner_index])

            # Уменьшение скорости обучения
            self.learning_rate -= 0.05

    def manhattan_distance(a, b):
        return np.sum(np.abs(a - b))

    def predict(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=1)
        return np.argmin(distances)