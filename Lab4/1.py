import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import classKohonenNN

data = pd.read_csv('students_data.csv', sep=';')  # Замените на путь к вашему файлу

features = data.iloc[:, 1:8]  # x1 - x7
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

kohonen_net = classKohonenNN.KohonenNetwork(input_size=7, output_size=4)
kohonen_net.train(normalized_features, epochs=9)

clusters = {}
for i in range(len(normalized_features)):
    cluster_id = kohonen_net.predict(normalized_features[i])
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(data.iloc[i]['x8'])  # x8 - информация о стипендии

for cluster_id, values in clusters.items():
    print(f'Cluster {cluster_id}: Average x8 value: {np.mean(values)}')