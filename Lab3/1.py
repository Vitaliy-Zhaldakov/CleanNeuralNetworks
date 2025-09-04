import numpy as np
import matplotlib.pyplot as plt
import classRBFN

# Заданные точки
S1_a = np.array([[2, 0], [1, 4], [1, 1]])  # Класс 1
S2_a = np.array([[3, 5], [1, 7], [7, 0]])  # Класс 0

S1_b = np.array([[0, 3], [-1, 4], [-2, -2], [-1, -4]])  # Класс 1
S2_b = np.array([[-1, -1], [0, 1], [-2, 1], [0, 0]])  # Класс 0

# Объединение данных
X_a = np.vstack((S1_a, S2_a))
y_a = np.array([1] * len(S1_a) + [0] * len(S2_a)).reshape(-1, 1)

X_b = np.vstack((S1_b, S2_b))
y_b = np.array([1] * len(S1_b) + [0] * len(S2_b)).reshape(-1, 1)

# Создание и обучение радиальной нейронной сети
rbf_network_a = classRBFN.RBFNetwork(n_centers=3, n_output=1)
rbf_network_a.fit(X_a, y_a)

rbf_network_b = classRBFN.RBFNetwork(n_centers=4, n_output=1)
rbf_network_b.fit(X_b, y_b)


# Функция для визуализации результатов
def plot_decision_boundary(rbf_network, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = rbf_network.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o')
    plt.title("Результат классификации")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Визуализация результатов для первого набора данных (a)
plot_decision_boundary(rbf_network_a, X_a, y_a)

# Визуализация результатов для второго набора данных (b)
plot_decision_boundary(rbf_network_b, X_b, y_b)