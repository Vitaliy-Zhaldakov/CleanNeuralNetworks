import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Lab4/students_data.csv', sep=';')

scaler = MinMaxScaler()
data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']] = scaler.fit_transform(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']])
input_data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']].values
koeff_data = data[['x8']].values

# Number of neurons in the SOM (4 clusters)
num_neurons = 4
# Input dimensionality (7 features)
input_dim = 7
# Learning rate
learning_rate = 0.30
# Neighborhood radius (for calculating weights)
neighborhood_radius = 2

# Initialize the weights randomly
weights = np.random.rand(num_neurons, input_dim)

# Training loop (6 epochs)
for epoch in range(6):
    # Update learning rate
    learning_rate -= 0.05
    # Shuffle the input data
    np.random.shuffle(input_data)

    # Iterate over each data point
    for i in range(len(input_data)):
        # Find the best-matching unit (BMU)
        #distances = np.linalg.norm(weights - input_data[i], axis=1)
        distances = np.sum(np.abs(weights - input_data[i]), axis=1)
        bmu_index = np.argmin(distances)

        # Update weights of the BMU and its neighbors
        for j in range(num_neurons):
            # Calculate the neighborhood distance
            distance_from_bmu = np.abs(j - bmu_index)
            if distance_from_bmu <= neighborhood_radius:
                # Update the weights using the learning rate and neighborhood distance
                weights[j] += learning_rate * (input_data[i] - weights[j]) * np.exp(
                    -distance_from_bmu ** 2 / (2 * neighborhood_radius ** 2))

# Assign clusters to data points
clusters = []
for i in range(len(input_data)):
  #distances = np.linalg.norm(weights - input_data[i], axis=1)
  distances = np.sum(np.abs(weights - input_data[i]), axis=1)
  cluster = np.argmin(distances)
  print(f'{cluster} - {koeff_data[i]}')
  clusters.append(cluster)
