import pickle
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo de mejores acciones
with open('best_actions.pkl', 'rb') as f:
    best_actions = pickle.load(f)

# Graficar el mapa de calor de mejores acciones
plt.figure(figsize=(10, 8))
plt.imshow(best_actions, cmap='plasma', origin='lower')

# Agregar etiquetas y título
plt.colorbar(label='Acción')
plt.title('Mejores Acciones en cada Estado')
plt.xlabel('Posición Y')
plt.ylabel('Posición X')
plt.xticks(np.arange(0, best_actions.shape[1]), np.arange(0, best_actions.shape[1]))
plt.yticks(np.arange(0, best_actions.shape[0]), np.arange(0, best_actions.shape[0]))

# Mostrar el gráfico
plt.show()
