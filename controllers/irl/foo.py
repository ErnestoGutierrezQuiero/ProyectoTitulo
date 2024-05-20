from gym.envs.registration import register

register(
    id='MiRobotGymEnv-v1',  # Identificador único para tu entorno
    entry_point='migym:MiRobotGymEnv',  # Apunta al módulo y clase
    max_episode_steps=100  # Define un límite de pasos por episodio (opcional)
)
