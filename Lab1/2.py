import numpy as np

class ModifiedWTANetwork:
    def __init__(self, num_neurons, learning_rate, win_threshold):
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.win_threshold = win_threshold
        self.weights = np.random.rand(num_neurons, 2)  # Инициализация весов случайными значениями
        self.wins = np.zeros(num_neurons)  # Счетчик побед для каждого нейрона

    def fit(self, X, y):
        for x in X:
            # Вычисляем активацию нейронов
            activations = np.dot(self.weights, x)
            # Находим индекс нейрона с максимальной активацией
            winner_index = np.argmax(activations)

            # Увеличиваем счетчик побед для победившего нейрона
            self.wins[winner_index] += 1

            # Применяем штраф к активации нейронов, которые превысили порог
            penalties = np.where(self.wins > self.win_threshold, 0.5 * (self.wins[winner_index] - self.win_threshold),
                                 0)
            activations -= penalties

            # Переопределяем победителя после применения штрафа
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


# Пример использования
if __name__ == "__main__":
    # Данные для обучения (двухкомпонентные векторы)
    X = np.array([[0.97, 0.2], [1.00, 0.00], [-0.72, 0.7], [-0.67, 0.74],
                  [-0.80, 0.6], [0.00, -1.00], [0.2, -0.97], [-0.3, -0.95],
                  [0.5, 0.97], [0.7, -0.87], [-0.2, 0.97], [0.6, -0.9]])
    # Метки классов (для примера)
    y = np.array([0, 1, 2, 3])  # Классы для каждого вектора

    # Создаем и обучаем сеть с порогом для штрафования
    modified_wta_network = ModifiedWTANetwork(num_neurons=4, learning_rate=0.5, win_threshold=2)
    modified_wta_network.fit(X, y)

    # Выводим веса нейронов после обучения
    print("Итоговые веса:")
    print(modified_wta_network.weights)

    # Пример предсказания
    test_vectors = np.array([[0.15, 0.25], [0.5, 0.5]])
    predictions = modified_wta_network.predict(test_vectors)
    print("Классификация на тестовых данных:")
    print(predictions)