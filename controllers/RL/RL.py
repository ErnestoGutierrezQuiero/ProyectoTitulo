from controller import Robot, Supervisor
import math
import numpy as np
import pickle

class MiRobotSupervisor:
    def __init__(self):
        self.robot = Robot()
        self.supervisor = Supervisor()
        self.motor_izq = self.robot.getDevice('left wheel motor')
        self.motor_der = self.robot.getDevice('right wheel motor')
        self.robot_node = self.supervisor.getFromDef("Pioneer")
        self.target_node = self.supervisor.getFromDef("target")
        self.robot_translation_field = self.robot_node.getField("translation")
        self.target_translation_field = self.target_node.getField("translation")
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.estadoAnterior = [0, 0]
        self.estado = [0, 0]
        self.x_min = 0
        self.x_max = 6
        self.y_min = 0
        self.y_max = 6
        self.num_boxes_x = 6
        self.num_boxes_y = 6
        self.box_width = (self.x_max - self.x_min) / self.num_boxes_x
        self.box_height = (self.y_max - self.y_min) / self.num_boxes_y
        self.sensores_delante = [
            self.robot.getDevice("ds14"),
            self.robot.getDevice("ds15"),
            self.robot.getDevice("ds0"),
            self.robot.getDevice("ds1")
        ]
        self.sensores_detras = [
            self.robot.getDevice("ds6"),
            self.robot.getDevice("ds7"),
            self.robot.getDevice("ds8"),
            self.robot.getDevice("ds9")
        ]
        for sensor in self.sensores_delante:
            sensor.enable(self.timestep)
        for sensor in self.sensores_detras:
            sensor.enable(self.timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.umbral = 1
        self.Q = np.zeros((self.num_boxes_x, self.num_boxes_y, 7))
        self.alpha = 0.16
        self.gamma = 0.7
        self.epsilon = 0.9
        self.epsilon_min=0.1
        self.decay=0.995
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        self.distance = np.linalg.norm(robot_position - target_position)
        self.distancia_anterior = 0
        pos = self.target_translation_field.getSFVec3f()
        x = pos[0]
        y = pos[1]
        self.target_state = self.discretizar(x, y)

    def discretizar(self, x, y):
        box_x = int((x - self.x_min) / self.box_width)
        box_y = int((y - self.y_min) / self.box_height)
        box_x = max(0, min(self.num_boxes_x - 1, box_x))
        box_y = max(0, min(self.num_boxes_y - 1, box_y))
        return box_x, box_y

    def state(self):
        pos = self.robot_translation_field.getSFVec3f()
        x = pos[0]
        y = pos[1]
        if not self.dentro_de_limites(x, y):
            raise ValueError("El robot está fuera de los límites permitidos")
        state = self.discretizar(x, y)
        self.estado = state
        return state

    def select_best_actions(self, state, num_actions=1):
        best_action_indices = np.argpartition(self.Q[state[0], state[1], :], -num_actions)[-num_actions:]
        return best_action_indices

    def SeleccionarAccion(self, estado):
        if np.random.rand() <= self.epsilon:
            accion = np.random.randint(7)
        else:
            best_actions = self.select_best_actions(estado, num_actions=2)
            accion = np.random.choice(best_actions)
        return accion

    def detener(self, tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(0)
        self.motor_der.setVelocity(0)
        self.robot.step(tiempo)

    def avanzar(self, tiempo):
        if self.puede_moverse_adelante():
            self.motor_izq.setPosition(float('inf'))
            self.motor_der.setPosition(float('inf'))
            self.motor_izq.setVelocity(6)
            self.motor_der.setVelocity(6)
            self.robot.step(tiempo)
        self.detener(1000)

    def retroceder(self, tiempo):
        if self.puede_moverse_detras():
            self.motor_izq.setPosition(float('inf'))
            self.motor_der.setPosition(float('inf'))
            self.motor_izq.setVelocity(-6)
            self.motor_der.setVelocity(-6)
            self.robot.step(tiempo)
        self.detener(1000)

    def girar_izquierda(self, tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(-1.7)
        self.motor_der.setVelocity(1.7)
        self.robot.step(tiempo)
        self.detener(1000)
        if self.puede_moverse_adelante():
            self.avanzar(2000)

    def girar_derecha(self, tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(1.7)
        self.motor_der.setVelocity(-1.7)
        self.robot.step(tiempo)
        self.detener(999)
        if self.puede_moverse_adelante():
            self.avanzar(2000)

    def girar_izquierda_retroceder(self, tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(1.7)
        self.motor_der.setVelocity(-1.7)
        self.robot.step(tiempo)
        self.detener(1000)
        if self.puede_moverse_detras():
            self.retroceder(2000)

    def girar_derecha_retroceder(self, tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(-1.7)
        self.motor_der.setVelocity(1.7)
        self.robot.step(tiempo)
        self.detener(1000)
        if self.puede_moverse_detras():
            self.retroceder(2000)

    def puede_moverse_adelante(self):
        for sensor in self.sensores_delante:
            if sensor.getValue() > self.umbral:
                return False
        return True

    def puede_moverse_detras(self):
        for sensor in self.sensores_detras:
            if sensor.getValue() > self.umbral:
                return False
        return True

    def recompensa(self):
        estado_actual = self.estado
        target_state = self.target_state
        distancia_actual = np.linalg.norm(np.array(target_state) - np.array(estado_actual))
        distancia_anterior = np.linalg.norm(np.array(target_state) - np.array(self.estadoAnterior))
        cambio_distancia = distancia_anterior - distancia_actual
        reward = 0
        if cambio_distancia > 0:
            reward += int(cambio_distancia * 40)
        else:
            reward += int(cambio_distancia * 10)
        if not self.puede_moverse_adelante() or not self.puede_moverse_detras():
            reward -= 20
        if self.estado == self.estadoAnterior:
            reward -= 1
        if estado_actual == target_state:
            reward += 1000
        return reward

    def cercania(self):
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        dif_pos = np.concatenate((robot_position, target_position))
        return dif_pos

    def reset(self):
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.estado = self.state()
        self.estadoAnterior = self.estado

    def done(self):
        robot_state = self.state()
        target_state = self.target_state
        if robot_state == target_state:
            return True
        else:
            return False

    def accion(self, opcion):
        self.estadoAnterior = self.state()
        if opcion == 0:
            self.detener(3000)
        elif opcion == 1:
            self.avanzar(2000)
        elif opcion == 2:
            self.girar_derecha(2000)
        elif opcion == 3:
            self.girar_izquierda(2000)
        elif opcion == 4:
            self.girar_derecha_retroceder(2000)
        elif opcion == 5:
            self.girar_izquierda_retroceder(2000)
        elif opcion == 6:
            self.retroceder(2000)
        self.estado = self.state()

    def step(self, accion):
        self.estadoAnterior = self.estado
        self.accion(accion)
        self.robot.step(self.timestep)
        estado = self.state()
        self.estado = estado
        rew = self.recompensa()
        fin = self.done()
        return estado, rew, fin

    def entrenar(self, episodios):
        recompensas = []
        for episodio in range(episodios):
            print(f"Episodio {episodio}  -", self.epsilon)
            self.reset()
            self.estado = self.state()
            self.estadoAnterior = (0, 0)
            recompensa_total = 0
            fin = False
            contador = 0
            while not fin:
                accion = self.SeleccionarAccion(self.estado)
                siguiente_estado, recompensa, fin = self.step(accion)
                siguiente_estado_discretizado = self.discretizar(siguiente_estado[0], siguiente_estado[1])
                siguiente_accion = self.SeleccionarAccion(siguiente_estado_discretizado)
                if not fin:
                    self.Q[self.estado[0], self.estado[1], accion] += self.alpha * (
                        recompensa +
                        self.gamma * self.Q[siguiente_estado_discretizado[0], siguiente_estado_discretizado[1], siguiente_accion] -
                        self.Q[self.estado[0], self.estado[1], accion])
                self.estado = siguiente_estado
                recompensa_total += recompensa
                contador += 1
                if contador >= 500:
                    fin = True
                    recompensa_total -= 100
            recompensas.append(recompensa_total)
            print(f'Fin del episodio {episodio}, recompensa total: {recompensa_total}')
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
        return recompensas

    def dentro_de_limites(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

rob = MiRobotSupervisor()
rewards = rob.entrenar(200)
print(rewards)
with open('q_table.pkl', 'wb') as f:
    pickle.dump(rob.Q, f)

best_actions = np.zeros((rob.num_boxes_x, rob.num_boxes_y), dtype=int)
for i in range(rob.num_boxes_x):
    for j in range(rob.num_boxes_y):
        best_actions[i, j] = np.argmax(rob.Q[i, j, :])

with open('best_actions.pkl', 'wb') as f:
    pickle.dump(best_actions, f)

print("Q-table and best actions have been saved to 'q_table.pkl' and 'best_actions.pkl'.")
