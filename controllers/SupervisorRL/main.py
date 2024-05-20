import gym
import numpy as np
import pickle
from entorno import WebotsGymEnv  # Importa el entorno Gym

# Crear el entorno Gym
env = WebotsGymEnv()

# Parámetros para SARSA
num_episodios = 50  # Número de episodios para entrenamiento
alpha = 0.5  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 0.1  # Tasa de exploración

# Crear la matriz Q
numEstados=[6,6,5]
Q = np.zeros((numEstados[0],numEstados[1], numEstados[2]))


for episodio in range(num_episodios):
    estado = env.reset()  
    done = False
    recompensa_total = 0
    steps = 0  # Contar

    while not done and steps < 100:  #nº pasos
         #epsilon greedy
        if np.random.rand() < epsilon:
            accion = np.random.randint(5)  # Exploración
        else:
            accion = np.argmax(Q[estado[0], estado[1], :])  # Explotación

        # Ejecutar la acción
        siguiente_estado, recompensa, done, _ = env.step(accion)
        
        # Actualizar la matriz Q
        if not done:
            siguiente_accion = np.argmax(Q[siguiente_estado[0], siguiente_estado[1], :])
            Q[estado[0], estado[1], accion] += alpha * (
                recompensa +
                gamma * Q[siguiente_estado[0], siguiente_estado[1], siguiente_accion] -
                Q[estado[0], estado[1], accion]
            )

        estado = siguiente_estado
        recompensa_total += recompensa
        steps += 1  # Incrementar el contador de pasos

    print(f'Episodio {episodio}, recompensa total: {recompensa_total}')


# Guardar la matriz Q en un archivo .pkl
with open("matriz_q.pkl", "wb") as f:
    pickle.dump(Q, f)

# Guardar las mejores acciones para cada estado en un archivo .pkl
mejores_acciones = np.argmax(Q, axis=2)  # Obtener las mejores acciones por estado
with open("mejores_acciones.pkl", "wb") as f:
    pickle.dump(mejores_acciones, f)
