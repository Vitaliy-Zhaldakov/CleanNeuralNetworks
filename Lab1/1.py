import numpy as np
import classWTANetwork as wta

# Пример использования
if __name__ == "__main__":
    # Данные для обучения (двухкомпонентные векторы)
    X = np.array([[0.97, 0.2], [1.00, 0.00], [-0.72, 0.7], [-0.67, 0.74],
                  [-0.80, 0.6], [0.00, -1.00], [0.2, -0.97], [-0.3, -0.95],
                  [0.5, 0.97], [0.7, -0.87], [-0.2, 0.97], [0.6, -0.9]])
    # Метки классов (для примера)
    y = np.array([0, 1, 2, 3])  # Классы для каждого вектора

    # Создаем и обучаем сеть
    wta_network = wta.WTANetwork(num_neurons=4, learning_rate=0.5)
    wta_network.fit(X, y)

    # Выводим веса нейронов после обучения
    print("Итоговые веса:")
    print(wta_network.weights)

    # Пример предсказания
    test_vectors = np.array([[0.15, 0.25], [0.5, 0.5]])
    predictions = wta_network.predict(test_vectors)
    print("Классификация на тестовых данных:")
    print(predictions)