import pickle

# Abrir el archivo pickle en modo de lectura binaria
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Ahora puedes acceder al contenido del objeto cargado
print("Tabla Q:")
print(q_table)

# Hacer lo mismo para el archivo 'best_actions.pkl'
with open('best_actions.pkl', 'rb') as f:
    best_actions = pickle.load(f)

# Imprimir el contenido del objeto cargado
print("Mejores acciones:")
print(best_actions)
