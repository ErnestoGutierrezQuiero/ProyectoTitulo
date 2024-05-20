from controller import Robot, Supervisor
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.2, gamma=0.9, epsilon=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.Q = np.zeros((num_states[0], num_states[1], num_actions))  # Inicializa la tabla Q

    def seleccionar_accion(self, estado):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Exploración
        else:
            return np.argmax(self.Q[estado[0], estado[1], :])  # Explotación

    def actualizar_Q(self, estado, accion, recompensa, siguiente_estado):
        mejor_siguiente_accion = np.argmax(self.Q[siguiente_estado[0], siguiente_estado[1], :])
        td_target = recompensa + self.gamma * self.Q[siguiente_estado[0], siguiente_estado[1], mejor_siguiente_accion]
        td_error = td_target - self.Q[estado[0], estado[1], accion]
        self.Q[estado[0], estado[1], accion] += self.alpha * td_error

class MiRobotSupervisor:
    def __init__(self):
        self.robot = Robot()
        self.supervisor = Supervisor()
        self.motor_izq = self.robot.getDevice('left wheel motor')
        self.motor_der = self.robot.getDevice('right wheel motor')
        self.robot_node=self.supervisor.getFromDef("Pioneer")
        self.target_node = self.supervisor.getFromDef("target")
        self.robot_translation_field = self.robot_node.getField("translation")
        self.target_translation_field = self.target_node.getField("translation")
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.estadoAnterior=[0,0]
        self.estado=[0,0]
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
            sensor.enable(self.timestep)  # Habilitar sensores
        for sensor in self.sensores_detras:
            sensor.enable(self.timestep) 
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(True)
        self.umbral = 1
        self.Q = np.zeros((self.num_boxes_x, self.num_boxes_y, 6))  # 6 acciones posibles
        self.alpha = 0.2#tasa aprendizaje
        self.gamma = 0.9  # factor descuento
        self.epsilon = 0.9  # exploracion    
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        self.distance = np.linalg.norm(robot_position - target_position)
        self.distancia_anterior=0
        pos = self.target_translation_field.getSFVec3f()  # Obtener la posición del objetivo
        x = pos[0]
        y = pos[1]
        self.target_state = self.discretizar(x, y) 
        self.agent = QLearningAgent((self.num_boxes_x, self.num_boxes_y), 7)

    def discretizar(self, x, y):
        # Convertir coordenadas continuas a índices discretos
        box_x = int((x - self.x_min) / self.box_width)
        box_y = int((y - self.y_min) / self.box_height)
        
        # verificar indice dentro de  los límites
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
        self.estado=state
        return state
        
    def select_best_actions(self, state, num_actions=1):
    # mejores acciones por estado
        best_action_indices = np.argpartition(self.Q[state[0], state[1], :], -num_actions)[-num_actions:]
        return best_action_indices
    
# Modify the SeleccionarAccion method to use select_best_actions
    def SeleccionarAccion(self, estado):
        if np.random.rand() <= self.epsilon:
            # Explore by choosing a random action
            accion = np.random.randint(6)
        else:
            # Exploit by selecting the best action
            best_action = np.argmax(self.Q[estado[0], estado[1], :])  # Select the single best action
            accion = best_action  # Choose the best action
        return accion

    
        
    def detener(self,tiempo):
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
        
    def girar_izquierda(self,tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(-1.7)
        self.motor_der.setVelocity(1.7)
        self.robot.step(tiempo)
        self.detener(1000)
        if self.puede_moverse_adelante():
        # Avanzar si es posible
            self.avanzar(2000)
            
            
    def girar_derecha(self,tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(1.7)
        self.motor_der.setVelocity(-1.7)
        self.robot.step(tiempo)
        self.detener(999)
        if self.puede_moverse_adelante():
        # Avanzar si es posible
            self.avanzar(2000)
            
    def girar_izquierda_retroceder(self, tiempo):
        # Girar a la izquierda
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(1.7)
        self.motor_der.setVelocity(-1.7)
        self.robot.step(tiempo)
        self.detener(1000)  # Detener después de girar

        # Retroceder
        if self.puede_moverse_detras():
            self.retroceder(2000)

    def girar_derecha_retroceder(self, tiempo):
        # Girar a la derecha
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(-1.7)
        self.motor_der.setVelocity(1.7)
        self.robot.step(tiempo)
        self.detener(1000)  # Detener después de girar

        # Retroceder
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
        estado_actual = self.estado  # Estado actual (discretizado)
        # Posición del objetivo
        target_state = self.target_state # Estado del objetivo (discretizado)        
        # Calcular la distancia al objetivo desde el estado actual y el estado anterior
        distancia_actual = np.linalg.norm(np.array(target_state) - np.array(estado_actual))
        distancia_anterior = np.linalg.norm(np.array(target_state) - np.array(self.estadoAnterior))        
        # Evaluar el cambio en la distancia
        cambio_distancia = distancia_anterior - distancia_actual  # Positivo si se acerca, negativo si se aleja     
        # Calcular la recompensa
        reward = 0        
        # Recompensa por acercarse al objetivo y penalización por alejarse
        if cambio_distancia > 0:
            reward += 50  # Recompensa por acercarse
        else:
            reward -= 15  # Penalización por alejarse            
        # Penalización por colisiones
        if not self.puede_moverse_adelante() or not self.puede_moverse_detras():
            reward -= 5 # Penalización por colisiones        
        # Penalización por no moverse
        if self.estado == self.estadoAnterior:
            reward -= 1  # Penalización por no moverse        
        # Recompensa significativa por alcanzar el objetivo
        if estado_actual == target_state:
            reward += 1500  # Recompensa por alcanzar el objetivo    
        return reward

    
    
            
    def cercania(self):
        # Obtener la posición actual del robot y del objetivo
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        dif_pos=np.concatenate((robot_position, target_position))
        return dif_pos

    def reset(self): #Resetear posicion y fisicas del entorno
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        e=self.state()
        self.estado=e
        self.estadoAnterior=e
    
    def done(self):
        robot_state = self.state()
        target_state = self.target_state # Obtener el estado discretizado del target
        if robot_state == target_state: # Verificar si el robot ha alcanzado el target
            return True  
        else:
            return False

    def accion(self, opcion):
        self.estadoAnterior=self.state()
        if opcion==0: 
            self.avanzar(2000)
        elif opcion==1:
            self.girar_derecha(2000)
        elif opcion==2:
            self.girar_izquierda(2000)
        elif opcion==3:
            self.girar_derecha_retroceder(2000)
        elif opcion==4:
            self.girar_izquierda_retroceder(2000)
        elif opcion==5:
            self.retroceder(2000)
        self.estado=self.state()
        
        
    def step(self, accion):
        # Actualizar el estado anterior antes de tomar una nueva acción
        self.estadoAnterior = self.estado  # Guardar el estado actual como el anterior
        # Realizar la acción
        self.accion(accion)        
        # Simular un paso
        self.robot.step(self.timestep)        
        # Obtener el nuevo estado
        estado = self.state()  # Obtener el estado actual después del paso
        self.estado=estado
        # Calcular la recompensa con el estado actualizado
        rew = self.recompensa()
        # Verificar si el episodio ha terminado
        fin = self.done()
        return estado, rew, fin
    

    def entrenar(self, episodios):
        recompensas = []
        for episodio in range(episodios):
            print(f"Episodio {episodio}")
            
            # Restablecer la simulación
            self.reset()
    
            # Estado inicial
            self.estado = self.state()
            self.estadoAnterior=(0, 0)
            recompensa_total = 0
            fin = False
            contador = 0
            
            while not fin:
                # Elegir acción basada en el estado actual
                accion = self.SeleccionarAccion(self.estado)
                # Realizar el paso y obtener el siguiente estado y recompensa
                siguiente_estado, recompensa, fin = self.step(accion)
                # Discretizar el siguiente estado
                siguiente_estado_discretizado = self.discretizar(siguiente_estado[0], siguiente_estado[1])
                # Seleccionar la siguiente acción para el siguiente estado (antes de actualizar Q)
                siguiente_accion = self.SeleccionarAccion(siguiente_estado_discretizado)  # Corregir aquí
    
                # Actualizar la matriz `Q` usando SARSA o método similar
                if not fin:
                    self.Q[self.estado[0], self.estado[1], accion] += self.alpha * (
                        recompensa +
                        self.gamma * self.Q[siguiente_estado_discretizado[0], siguiente_estado_discretizado[1], siguiente_accion] -
                        self.Q[self.estado[0], self.estado[1], accion])
                # Actualizar el estado actual y la acción
                self.estado = siguiente_estado
                recompensa_total += recompensa
                contador += 1
                if contador >= 100:
                    fin = True
                    recompensa_total -= 50  # Penalización por episodios largos
            recompensas.append(recompensa_total)  # Guardar la recompensa total del episodio
            print(f'Fin del episodio {episodio}, recompensa total: {recompensa_total}')
            if self.epsilon > 0.1:
                self.epsilon -= 0.02
                self.epsilon = max(0.1, self.epsilon)
        return recompensas
    
    def dentro_de_limites(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def entrenarQL(self, episodios):
        recompensas = []
        for episodio in range(episodios):
            print(f"Episodio {episodio}")

            # Restablecer la simulación
            self.reset()

            # Estado inicial
            self.estado = self.state()
            self.estadoAnterior = (0, 0)
            recompensa_total = 0
            fin = False
            contador = 0

            while not fin:
                # Elegir acción basada en el estado actual
                accion = self.agent.seleccionar_accion(self.estado)
                # Realizar el paso y obtener el siguiente estado y recompensa
                siguiente_estado, recompensa, fin = self.step(accion)
                # Actualizar la matriz `Q` usando Q-learning
                self.agent.actualizar_Q(self.estado, accion, recompensa, siguiente_estado)
                # Actualizar el estado actual y la acción
                self.estado = siguiente_estado
                recompensa_total += recompensa
                contador += 1
                if contador >= 100:
                    fin = True
                    recompensa_total -= 200  # Penalización por episodios largos
            recompensas.append(recompensa_total)  # Guardar la recompensa total del episodio
            print(f'Fin del episodio {episodio}, recompensa total: {recompensa_total}')
            if self.agent.epsilon > 0.1:
                self.agent.epsilon -= 0.02
                self.agent.epsilon = max(0.1, self.agent.epsilon)
        return recompensas

            
rob = MiRobotSupervisor()
rewards = rob.entrenar(50)
r= rob.entrenarQL(50)
print(rewards)
# Save the Q-table to a pickle file
with open('q_table.pkl', 'wb') as f:
    pickle.dump(rob.Q, f)

# Create a table of best actions for every state
best_actions = np.zeros((rob.num_boxes_x, rob.num_boxes_y), dtype=int)
for i in range(rob.num_boxes_x):
    for j in range(rob.num_boxes_y):
        best_actions[i, j] = np.argmax(rob.Q[i, j, :])

# Save the best actions table to a pickle file
with open('best_actions.pkl', 'wb') as f:
    pickle.dump(best_actions, f)

print("Q-table and best actions have been saved to 'q_table.pkl' and 'best_actions.pkl'.")

plt.plot(rewards)
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.show()

plt.plot(r)
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.show()