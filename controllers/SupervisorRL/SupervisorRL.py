from controller import Robot, Supervisor
import math
import pandas as pd
import numpy as np
import time
import schedule
import time
import subprocess
import socket
import signal
import pickle

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
            self.robot.getDevice("ds13"),
            self.robot.getDevice("ds14"),
            self.robot.getDevice("ds15"),
            self.robot.getDevice("ds0"),
            self.robot.getDevice("ds1"),
            self.robot.getDevice("ds2")
        ]
        self.sensores_detras = [
            self.robot.getDevice("ds5"),
            self.robot.getDevice("ds6"),
            self.robot.getDevice("ds7"),
            self.robot.getDevice("ds8"),
            self.robot.getDevice("ds9"),
            self.robot.getDevice("ds10")
        ]
        for sensor in self.sensores_delante:
            sensor.enable(True)  # Habilitar los sensores
        for sensor in self.sensores_detras:
            sensor.enable(True) 
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(True)
        self.umbral = 2.0
        self.Q = np.zeros((self.num_boxes_x, self.num_boxes_y, 5))  # 5 acciones posibles
        self.alpha = 0.5#learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        self.distance = np.linalg.norm(robot_position - target_position)
        self.distancia_anterior=0
    
    def discretizar(self, x, y):
        # Convertir coordenadas continuas a índices discretos
        box_x = int((x - self.x_min) / self.box_width)
        box_y = int((y - self.y_min) / self.box_height)
        
        # Asegurarse de que los índices no excedan los límites
        box_x = max(0, min(self.num_boxes_x - 1, box_x))
        box_y = max(0, min(self.num_boxes_y - 1, box_y))
        
        return box_x, box_y
        
    def state(self):
        pos = self.robot_translation_field.getSFVec3f()
        x = pos[0]
        y = pos[1]
    
        if not self.dentro_de_limites(x, y):
            raise ValueError("El robot está fuera de los límites permitidos")
    
        # Discretizar el estado
        state = self.discretizar(x, y)
    
        return state
        
    def SeleccionarAccion(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(5)  # Elegir una acción aleatoria
        else:
            return np.argmax(self.Q[state[0], state[1], :])  # Elegir la mejor acción
   
    def target_state(self):
        pos = self.target_translation_field.getSFVec3f()  # Obtener la posición del objetivo
        x = pos[0]
        y = pos[1]
        target_state = self.discretizar(x, y)  # Discretizar la posición del objetivo
        return target_state

        
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
        self.detener(500)
        
        
    def retroceder(self, tiempo):
        if self.puede_moverse_detras():
            self.motor_izq.setPosition(float('inf'))
            self.motor_der.setPosition(float('inf'))
            self.motor_izq.setVelocity(-6)
            self.motor_der.setVelocity(-6)
            self.robot.step(tiempo)
        self.detener(500)
        
    def girar_izquierda(self,tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(-1.7)
        self.motor_der.setVelocity(1.7)
        self.robot.step(tiempo)
        self.detener(500)
        
    def girar_derecha(self,tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(1.7)
        self.motor_der.setVelocity(-1.7)
        self.robot.step(tiempo)
        self.detener(500)

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
    
    def recompensa(self, estado):
    # Lógica para calcular la recompensa
        self.distancia_anterior=self.distance
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        self.distance = np.linalg.norm(robot_position - target_position)
        
        if self.distance < 0.1:
            return 100  # Recompensa por alcanzar el objetivo
        elif not self.puede_moverse_adelante() or not self.puede_moverse_detras():
            return -1  # Penalización por chocar con un obstáculo
        elif self.distance > self.distancia_anterior:
            return -0.5  # Penalización por alejarse del objetivo
        elif self.distance < self.distancia_anterior:
            return 5  # Recompensa por acercarse
        else:
            return 0

            
            
    def cercania(self):
        # Obtener la posición actual del robot y del objetivo
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        dif_pos=np.concatenate((robot_position, target_position))
        return dif_pos

    def reset(self):
        self.supervisor.simulationResetPhysics()
        self.supervisor.simulationReset()
    
    def done(self):
        robot_state = self.state()
        target_state = self.target_state()  # Obtener el estado discretizado del target
    
        # Verificar si el robot ha alcanzado el target
        if robot_state == target_state:
            return True  # El episodio ha terminado
        else:
            return False  # El episodio continúa

    def accion(self, opcion):
        self.estadoAnterior=self.state()
        if opcion==0:
            self.detener(2000)
        elif opcion==1:    
            self.avanzar(2000)
        elif opcion==2:
            self.girar_derecha(2000)
        elif opcion==3:
            self.girar_izquierda(2000)
        elif opcion==4:
            self.retroceder(2000)
        self.Estado=self.state()
        
        
    def step(self, accion):
        self.accion(accion)
        self.robot.step(self.timestep)
        estado = self.state()  # Obtener el estado actual

        # Llamar a recompensa con el estado correcto
        rew = self.recompensa(estado)  # Pasar el estado como argumento
        
        fin = self.done()  # Verificar si el episodio ha terminado
        
        return estado, rew, fin
    

    def entrenar(self, episodios):
        recompensas = []
        for episodio in range(episodios):
            print(f"Episodio {episodio}")
    
            # Restablecer la simulación
            self.reset()
    
            # Estado inicial
            self.estado = self.state()
    
            recompensa_total = 0
            fin = False
            contador = 0
            
            while not fin:
                # Elegir acción basada en el estado actual
                accion = self.SeleccionarAccion(self.estado)
                # Realizar el paso y obtener el siguiente estado y recompensa
                siguiente_estado, recompensa, fin = self.step(accion)
                print(self.distance , self.distancia_anterior)
                print("R: ", recompensa)
                # Discretizar el siguiente estado
                siguiente_estado_discretizado = self.discretizar(siguiente_estado[0], siguiente_estado[1])
                # Seleccionar la siguiente acción para el siguiente estado (antes de actualizar Q)
                siguiente_accion = self.SeleccionarAccion(siguiente_estado_discretizado)  # Corregir aquí
    
                # Actualizar la matriz `Q` usando SARSA o método similar
                if not fin:
                    self.Q[self.estado[0], self.estado[1], accion] += self.alpha * (
                        recompensa +
                        self.gamma * self.Q[siguiente_estado_discretizado[0], siguiente_estado_discretizado[1], siguiente_accion] -
                        self.Q[self.estado[0], self.estado[1], accion]
                    )
    
                # Actualizar el estado actual y la acción
                self.estado = siguiente_estado
                recompensa_total += recompensa
    
                contador += 1
    
                if contador >= 200:
                    fin = True
                    recompensa_total -= 100  # Penalización por episodios largos
    
            recompensas.append(recompensa_total)  # Guardar la recompensa total del episodio
            print(f'Fin del episodio {episodio}, recompensa total: {recompensa_total}')
    
        return recompensas
    
    def dentro_de_limites(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

            
rob = MiRobotSupervisor()
print(rob.target_state()) 
rewards = rob.entrenar(20)

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