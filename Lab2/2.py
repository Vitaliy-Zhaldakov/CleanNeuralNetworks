import numpy as np

# Функция активации ReLU
def relu(x):
    return np.maximum(0, x)

# Производная функции активации ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Функция активации сигмоиды
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации сигмоиды
def sigmoid_derivative(x):
    return x * (1 - x)

# Генерация данных для задачи XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])  # Ожидаемые выходы для XOR

# Инициализация весов
np.random.seed(42)  # Для воспроизводимости
input_size = 2
hidden_size = 2
output_size = 1

weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

# Параметры обучения
learning_rate = 0.1
epochs = 10000

# Обучение модели
for epoch in range(epochs):
    # Прямое распространение
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Вычисление ошибки
    error = y - predicted_output

    # Обратное распространение ошибки
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_output)

    # Обновление весов
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Предсказания после обучения
hidden_layer_input = np.dot(X, weights_input_hidden)
hidden_layer_output = relu(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

# Вывод предсказаний
predictions = (predicted_output > 0.5).astype(int)
print("Предсказания:")
for i in range(len(X)):
    print(f"Вход: {X[i]}, Предсказание: {predictions[i][0]}")
