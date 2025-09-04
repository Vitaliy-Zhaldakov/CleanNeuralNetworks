import numpy as np

class RBFNetwork:
    def __init__(self, n_centers, n_output):
        self.n_centers = n_centers
        self.n_output = n_output
        self.centers = None
        self.weights = None

    def _gaussian(self, x, center, width=1):
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * width**2))

    def _calculate_activations(self, X):
        activations = np.zeros((X.shape[0], self.n_centers))
        for i in range(self.n_centers):
            for j in range(X.shape[0]):
                activations[j, i] = self._gaussian(X[j], self.centers[i])
        return activations

    def fit(self, X, y):
        # Инициализация центров как случайные точки из данных
        random_indices = np.random.choice(X.shape[0], self.n_centers, replace=False)
        self.centers = X[random_indices]

        # Вычисление активаций
        activations = self._calculate_activations(X)

        # Обучение весов с помощью псевдообратной матрицы
        self.weights = np.linalg.pinv(activations).dot(y)

    def predict(self, X):
        activations = self._calculate_activations(X)
        return np.dot(activations, self.weights)