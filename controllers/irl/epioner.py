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
        self.sensores_delante = [
            self.robot.getDevice("ds14"),
            self.robot.getDevice("ds15"),
            self.robot.getDevice("ds0"),
            self.robot.getDevice("ds1"),
        ]
        self.sensores_detras = [
            self.robot.getDevice("ds6"),
            self.robot.getDevice("ds7"),
            self.robot.getDevice("ds8"),
            self.robot.getDevice("ds9"),
        ]
        for sensor in self.sensores_delante:
            sensor.enable(True)  # Habilitar los sensores
        for sensor in self.sensores_detras:
            sensor.enable(True) 
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(True)
        self.umbral = 2.0
        self.Q = np.zeros([6, 6, 5])
        self.alpha = 0.5  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        
    def state(self):
        pos= self.robot_translation_field.getSFVec3f()
        x= pos[0]
        y=pos[1]
        stat=[int(x),int(y)]
        self.estado=[int(x),int(y)]
        return stat
    
    def SeleccionarAccion(self, state):
        if np.random.rand()<=self.epsilon:
            return np.random.randint(5)
        else:
            return np.argmax(self.Q[state[0],state[1],:])     
    
    def seleccionarAccionFeedback(self, estado, entrenador, feedbackProbabilidad):
        if np.random.rand() <= feedbackProbabilidad: #consejo
            if np.random.rand() <= self.calidad: #buen consejo 
                return np.argmax(self.Q[estado[0], estado[1],estado[2], :])
            else:
                return np.argmin(entrenador.Q[estado[0], estado[1],estado[2], :])
        else:#accion agente
            return self.SeleccionarAccion(estado)
    
    def detener(self):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(0)
        self.motor_der.setVelocity(0)
           
    def avanzar(self, tiempo):
        if self.puede_moverse_adelante():
            self.motor_izq.setPosition(float('inf'))
            self.motor_der.setPosition(float('inf'))
            self.motor_izq.setVelocity(5)  # Reducir velocidad
            self.motor_der.setVelocity(5)  # Reducir velocidad
            self.robot.step(tiempo)  # Esperar para permitir que el robot avance
        self.detener()
        
        
    def retroceder(self, tiempo):
        if self.puede_moverse_detras():
            self.motor_izq.setPosition(float('inf'))
            self.motor_der.setPosition(float('inf'))
            self.motor_izq.setVelocity(-5)
            self.motor_der.setVelocity(-5)
            self.robot.step(tiempo)
        self.detener()
        
    def girar_izquierda(self,tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(-1)
        self.motor_der.setVelocity(1)
        self.robot.step(tiempo)
        self.detener()
        
    def girar_derecha(self,tiempo):
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))
        self.motor_izq.setVelocity(1)
        self.motor_der.setVelocity(-1)
        self.robot.step(tiempo)
        self.detener()

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
    
    def recompensa(self,state):
        distanciaAnterior=math.sqrt((14-self.estadoAnterior[0])*(14-self.estadoAnterior[0])+(15-self.estadoAnterior[1])*(15-self.estadoAnterior[1]))
        distanciaActual=math.sqrt((14-self.estado[0])*(14-self.estado[0])+(15-self.estado[1])*(15-self.estado[1]))
        if(self.puede_moverse_adelante==False or self.puede_moverse_detras==False): #Cuando va a chocar con un objeto
            return -20
        elif(self.estado==self.estadoAnterior):#Cuando se aleja del objetivo final
            return -1
        elif(self.estado[1]>9):#Cuando llega al objetivo final
            return 10
        elif((distanciaAnterior-distanciaActual)>0):#Cuando se acerca al objetivo final
            return 1.5
        return -1 #Recompensa en caso de no caer en ningún bloque if
    
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
        robot_position = np.array(self.robot_translation_field.getSFVec3f())
        target_position = np.array(self.target_translation_field.getSFVec3f())
        distance_to_target = np.linalg.norm(robot_position - target_position)
        if distance_to_target < 0.1:  # Define una distancia de proximidad al objetivo
            done = True  # Termina el episodio si el robot alcanza el objetivo
        else:
            done = False  # Continúa el episodio si el robot no ha alcanzado el objetivo
        return done

    def accion(self, opcion):
        self.estadoAnterior=self.state()
        if opcion==0:
            self.detener()
        elif opcion==1:    
            self.avanzar(1000)
        elif opcion==2:
            self.girar_derecha(1000)
        elif opcion==3:
            self.girar_izquierda(1000)
        elif opcion==4:
            self.retroceder(1000)
        self.Estado=self.state()
        
        
    def step(self,accion):
        if accion==1:
            if self.puede_moverse_adelante:
                self.accion(accion)
            else:
                self.accion(0)
        elif accion==4:
            if self.puede_moverse_detras:
                self.accion(accion)
            else:
                self.accion(0)
        else:
            self.accion(accion)
        self.robot.step(2000)
        estado=self.state()
        rew=self.recompensa(estado)
        fin = self.done()       
        return estado, rew, fin
    
    def entrenar(self, episodios, entrenador=None, feedbackProbabilidad=0, rShapping=0):   
        recompensas = []
        for episodio in range(episodios):
            print(f"Episodio {episodio}")
            self.reset()  # Reinicia el entorno y la física del robot
            estado = self.state()  # Obtiene el estado inicial
            recompensa_total = 0
            fin = False
            contador = 0  # Para evitar bucles infinitos o episodios muy largos    
            while not fin:
                # Selecciona una acción basada en feedback o política epsilon-greedy
                accion = self.seleccionarAccionFeedback(estado, entrenador, feedbackProbabilidad)
                # Ejecuta la acción y recibe el siguiente estado, recompensa, y si el episodio ha terminado
                siguiente_estado, recompensa, fin = self.step(accion)
                # Selecciona la siguiente acción para SARSA
                siguiente_accion = self.SeleccionarAccion(siguiente_estado)
                # Aumenta el total de recompensas
                recompensa_total += recompensa
                # Si el contador alcanza un límite, termina el episodio para evitar bucles infinitos
                if contador >= 500:
                    fin = True
                    recompensa = -200  # Penalización para episodios largos
                # Si el episodio no ha terminado, actualiza la matriz Q utilizando SARSA
                if not fin:
                    # Actualiza la tabla Q utilizando SARSA
                    self.Q[estado[0], estado[1], accion] += self.alpha * (recompensa + self.gamma * self.Q[siguiente_estado[0], siguiente_estado[1], siguiente_accion] - self.Q[estado[0], estado[1], accion])
                estado = siguiente_estado  # Avanza al siguiente estado
                accion = siguiente_accion  # Actualiza la acción para SARSA
                contador += 1  # Incrementa el contador
                print(recompensa_total)
            print(f'Fin del episodio {episodio}, recompensa total: {recompensa_total}')
            recompensas.append(recompensa_total)  # Guarda la recompensa total del episodio
        return recompensas