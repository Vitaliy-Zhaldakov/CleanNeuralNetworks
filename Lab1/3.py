import numpy as np

class HebbianNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))  # Инициализация весов нулями

    def fit(self, X):
        for x in X:
            # Применяем правило Хебба для обновления весов
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if i != j:  # Не обновляем вес самосвязи
                        self.weights[i][j] += x[i] * x[j]

    def predict(self, X):
        predictions = []
        for x in X:
            activations = np.dot(self.weights, x)
            predictions.append(np.argmax(activations))
        return np.array(predictions)

# Пример использования
if __name__ == "__main__":
    # Данные для обучения (двухкомпонентные векторы)
    X = np.array([[0.97, 0.2], [1.00, 0.00], [-0.72, 0.7], [-0.67, 0.74],
                  [-0.80, 0.6], [0.00, -1.00], [0.2, -0.97], [-0.3, -0.95],
                  [0.5, 0.97], [0.7, -0.87], [-0.2, 0.97], [0.6, -0.9]])  # Примеры входных векторов
    # Создаем и обучаем сеть
    hebbian_network = HebbianNetwork(num_neurons=2)
    hebbian_network.fit(X)

    # Выводим веса нейронов после обучения
    print("Итоговые веса:")
    print(hebbian_network.weights)

    # Пример предсказания
    test_vectors = np.array([[1, 0], [0, 1]])
    predictions = hebbian_network.predict(test_vectors)
    print("Классификация на тестовых данных:")
    print(predictions)