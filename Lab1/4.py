import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Инициализация весов и смещения
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Обучение перцептрона
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Обновление весов и смещения
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

# Установка семени генератора случайных чисел для воспроизводимости
np.random.seed(42)

# Генерация 20 случайных точек в положительном квадранте (0 <= x1, x2 <= 1)
num_points = 20
X_train = np.random.rand(num_points, 2)

# Присвоение меток: 1 если x1 > x2, иначе 0
y_train = (X_train[:, 0] > X_train[:, 1]).astype(int)

# Обучение перцептрона на тренировочных данных
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)

# Генерация тестовых данных (1000 точек)
num_test_points = 100
X_test = np.random.rand(num_test_points, 2)
y_test = (X_test[:, 0] > X_test[:, 1]).astype(int)

# Прогнозирование на тестовых данных
predictions = perceptron.predict(X_test)

# Вычисление точности
accuracy = np.mean(predictions == y_test)
print(f'Точность модели на тестовых данных: {accuracy:.2f}')

# Визуализация тестовых данных и предсказаний
plt.scatter(X_test[predictions == 1][:, 0], X_test[predictions == 1][:, 1], color='blue', label='Предсказанный положительный класс')
plt.scatter(X_test[predictions == 0][:, 0], X_test[predictions == 0][:, 1], color='red', label='Предсказанный отрицательный класс')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Результат работы сети')
plt.legend()
plt.grid()
plt.show()