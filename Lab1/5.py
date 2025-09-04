import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Инициализация весов и смещения
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Обучение
        for _ in range(self.n_iter):
            linear_output = np.dot(X, self.weights) + self.bias
            errors = y - linear_output

            # Обновление весов и смещения с использованием метода наименьших квадратов
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * np.sum(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)


# Установка случайного зерна для воспроизводимости
np.random.seed(42)

# Генерация 20 случайных точек в положительном квадранте
n_train = 20
X_train = np.random.rand(n_train, 2)  # Две координаты (x1, x2)

# Присвоение меток: 1 если x1 > x2, иначе 0
y_train = (X_train[:, 0] > X_train[:, 1]).astype(int)

# Создание DataFrame для удобства
train_data = pd.DataFrame(X_train, columns=['x1', 'x2'])
train_data['label'] = y_train

print("Тренировочный набор данных:")
print(train_data)

# Обучение нейрона Адалайн на тренировочных данных
adaline = Adaline(learning_rate=0.01, n_iter=1000)
adaline.fit(X_train, y_train)

# Генерация тестового набора данных из 1000 точек
n_test = 100
X_test = np.random.rand(n_test, 2)
y_test = (X_test[:, 0] > X_test[:, 1]).astype(int)

# Предсказание на тестовых данных
predictions = adaline.predict(X_test)

# Оценка точности
accuracy = np.mean(predictions == y_test)
print(f"Точность модели на тестовом наборе: {accuracy:.2f}")

# Визуализация результатов
plt.scatter(X_test[predictions == 1][:, 0], X_test[predictions == 1][:, 1], color='blue', label='Предсказанный положительный класс')
plt.scatter(X_test[predictions == 0][:, 0], X_test[predictions == 0][:, 1], color='red', label='Предсказанный отрицательный класс')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Результат работы сети (Нейрон типа Адалайн)')
plt.legend()
plt.grid()
plt.show()
