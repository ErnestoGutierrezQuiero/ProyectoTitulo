import pickle
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos de recompensas desde el archivo pkl
with open('recompensas.pkl', 'rb') as f:
    rewards = pickle.load(f)

# Convertir la lista de recompensas a un array de numpy para facilitar el cálculo
rewards_array = np.array(rewards)

# Calcular el rango intercuartil (IQR)
Q1 = np.percentile(rewards_array, 25)
Q3 = np.percentile(rewards_array, 75)
IQR = Q3 - Q1

# Definir un umbral para identificar outliers
umbral = 1.5

# Calcular los límites inferior y superior para identificar outliers
limite_inferior = Q1 - umbral * IQR
limite_superior = Q3 + umbral * IQR

# Filtrar las recompensas para eliminar outliers
rewards_filtradas = [r for r in rewards if limite_inferior <= r <= limite_superior]

# Aplicar suavizado a las recompensas filtradas
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Calcular el promedio móvil con un tamaño de ventana de 10 episodios
window_size = 10
smoothed_rewards = moving_average(rewards_filtradas, window_size)

# Graficar las recompensas suavizadas
plt.plot(smoothed_rewards)
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total Suavizada')
plt.title(f'Recompensas Suavizadas con Promedio Móvil (Tamaño de Ventana = {window_size})')
plt.show()
