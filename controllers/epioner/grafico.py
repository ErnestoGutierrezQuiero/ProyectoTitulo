import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Definir las etiquetas para la barra de color
acciones_etiquetas = ['Avanzar', 'Girar Derecha y Avanzar', 'Girar Izquierda y Avanzar', 'Girar Derecha y Retroceder', 'Girar Izquierda y Retroceder', 'Retroceder']

# Cargar el archivo de mejores acciones
with open('best_actions.pkl', 'rb') as f:
    best_actions = pickle.load(f)

# Definir un mapa de colores personalizado
cmap = mcolors.ListedColormap(['red', 'blue', 'green', 'yellow', 'purple', 'orange'])
bounds = np.arange(7) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Graficar el mapa de calor de mejores acciones
plt.figure(figsize=(10, 8))
cax = plt.imshow(best_actions, cmap=cmap, norm=norm, origin='lower')

# Agregar etiquetas y título
cbar = plt.colorbar(cax, ticks=np.arange(6))
cbar.set_label('Acción')
cbar.set_ticklabels(acciones_etiquetas)  # Asignar las etiquetas definidas anteriormente

plt.title('Mejores Acciones en cada Estado')
plt.xlabel('Posición Y')
plt.ylabel('Posición X')
plt.xticks(np.arange(0, best_actions.shape[1]), np.arange(0, best_actions.shape[1]))
plt.yticks(np.arange(0, best_actions.shape[0]), np.arange(0, best_actions.shape[0]))

# Añadir los valores de las acciones en cada celda
for i in range(best_actions.shape[0]):
    for j in range(best_actions.shape[1]):
        plt.text(j, i, str(best_actions[i, j]), ha='center', va='center', color='black', fontsize=12)

# Mostrar el gráfico
plt.show()
