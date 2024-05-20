# Paso 1: Definir el entorno Gym
import numpy as np
from epioner import MiRobotSupervisor
from gym import spaces
import gym

class MiRobotGymEnv(gym.Env):
    def __init__(self):
        # Paso 2: Inicializar el entorno utilizando tu clase existente
        self.robot = MiRobotSupervisor()

        # Paso 3: Definir el espacio de acción
        # Por ejemplo, 5 acciones: detener, avanzar, girar derecha, girar izquierda, retroceder
        self.action_space = spaces.Discrete(5)

        # Paso 4: Definir el espacio de observación
        # Un espacio de observación para una posición 2D (x, y)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self):
        # Reiniciar el entorno
        self.robot.reset()  # Reiniciar el robot
        return self.robot.state()  # Devolver el estado inicial

    def step(self, action):
        if action==1:
            if self.robot.puede_moverse_adelante():
        # Ejecutar la acción y obtener el siguiente estado, recompensa, y si el episodio terminó
                next_state, reward, done = self.robot.step(action)  # Usa tu método `step`
        elif action==4:
            if self.robot.puede_moverse_detras():
                next_state, reward, done = self.robot.step(action)  # Usa tu método `step`
        else:
            next_state, reward, done = self.robot.step(action)  # Usa tu método `step`
        return next_state, reward, done, {}  # Regresar valores compatibles con Gym

    def render(self, mode='human'):
        # Este método es opcional y puede ser utilizado para renderizar el entorno
        pass

    def close(self):
        # Limpieza y finalización del entorno (opcional)
        pass
