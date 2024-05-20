import gym
import numpy as np
from SupervisorRL import MiRobotSupervisor  # Importa el controlador

class WebotsGymEnv(gym.Env):
    def __init__(self):
        self.controlador = MiRobotSupervisor()
        self.action_space = gym.spaces.Discrete(5)  # 5 acciones
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([6, 6], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self):
        self.controlador.reset()  # Restablecer el controlador
        estado_inicial = self.controlador.state()  # Obtener el estado inicial
        return estado_inicial

    def step(self, action):
        # Aplicar la acción y avanzar la simulación
        self.controlador.accion(action)
        self.controlador.robot.step(self.controlador.timestep)

        # Obtener la recompensa y el estado actual
        recompensa = self.controlador.recompensa()
        estado = self.controlador.state()

        # Verificar si el episodio ha terminado (al llegar a 5, 5)
        done = (estado[0] == 5 and estado[1] == 5)

        return estado, recompensa, done, {}
