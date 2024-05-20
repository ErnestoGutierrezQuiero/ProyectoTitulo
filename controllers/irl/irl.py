
import gym
from gym import spaces
import numpy as np
import math
import foo
from mushroom_rl.algorithms.value.td import SARSA
from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils import dataset 
from gym.envs.registration import register

# Paso 5: Crear el entorno Gym usando la clase personalizada
env_name = 'MiRobotGymEnv-v1'
if not isinstance(env_name, str):
    raise TypeError("Expected 'env_name' to be a string")
else:
    print("tabien")
mirobot_env = Gym(gym.make('MiRobotGymEnv-v1'),100,0.99)  # Asegúrate de que 'env_name' sea una cadena

# Ejemplo de concatenación segura
info_str = "Environment Information: " + str(mirobot_env.info)
# Paso 6: Definir el agente SARSA con parámetros apropiados
agent = SARSA(
    mirobot_env.info,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.1,
)

# Paso 7: Crear el núcleo para entrenar el agente
core = Core(agent, mirobot_env)

# Paso 8: Entrenar por un número de pasos o episodios
dataset = core.learn(n_steps=150, n_steps_per_fit=10)

# Paso 9: Analizar resultados (recompensas acumuladas)
J = compute_J(dataset)

print("Recompensas acumuladas:", J)