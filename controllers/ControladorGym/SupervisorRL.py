import numpy as np
import math
from controller import Robot, Supervisor

class MiRobotSupervisor:
    def __init__(self):
        self.robot = Robot()  # Solo una instancia de Robot
        self.supervisor = Supervisor()  # Supervisor para controlar la simulación

        # Configuración de motores
        self.motor_izq = self.robot.getDevice('left wheel motor')
        self.motor_der = self.robot.getDevice('right wheel motor')
        self.motor_izq.setPosition(float('inf'))  # Posición infinita para control de velocidad
        self.motor_der.setPosition(float('inf'))

        # Configuración de sensores y campos
        self.robot_node = self.supervisor.getFromDef("Pioneer")
        self.target_node = self.supervisor.getFromDef("target")
        self.robot_translation_field = self.robot_node.getField("translation")
        self.target_translation_field = self.target_node.getField("translation")

        # Habilitar sensores
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
            sensor.enable(True)  # Habilitar sensores con el timestep
        for sensor in self.sensores_detras:
            sensor.enable(True)


        # Parámetros generales
        self.timestep = int(self.supervisor.getBasicTimeStep())  # Tiempo de paso
        self.umbral = 2.5  # Umbral para detectar obstáculos

        # Parámetros para aprendizaje por refuerzo
        self.num_boxes_x = 6
        self.num_boxes_y = 6
        self.alpha = 0.5  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        self.epsilon = 0.1  # Tasa de exploración

    def discretizar(self, x, y):
        # Convertir coordenadas a índices discretos
        box_x = max(0, min(self.num_boxes_x - 1, int(x)))  # Limitar a los límites
        box_y = max(0, min(self.num_boxes_y - 1, int(y)))
        return box_x, box_y

    def state(self):
        # Obtener la posición actual del robot y discretizarla
        pos = self.robot_translation_field.getSFVec3f()
        x = pos[0]
        y = pos[1]
        return self.discretizar(x, y)  # Devolver estado discretizado

    def accion(self, accion_id):
        # Establecer posición infinita para motores antes de ajustar velocidad
        self.motor_izq.setPosition(float('inf'))
        self.motor_der.setPosition(float('inf'))

        # Realiza la acción en función del identificador
        if accion_id == 0:  # Detenerse
            self.motor_izq.setVelocity(0)
            self.motor_der.setVelocity(0)
        elif accion_id == 1:
            if self.puede_moverse_adelante():  # Avanzar
                self.motor_izq.setVelocity(10)
                self.motor_der.setVelocity(10)
        elif accion_id == 2:  # Girar izquierda
            self.motor_izq.setVelocity(-1)
            self.motor_der.setVelocity(1)
        elif accion_id == 3:  # Girar derecha
            self.motor_izq.setVelocity(1)
            self.motor_der.setVelocity(-1)
        elif accion_id == 4: 
            if self.puede_moverse_detras():  # Retroceder
                self.motor_izq.setVelocity(-10)
                self.motor_der.setVelocity(-10)
        # Avanzar el paso de simulación para ejecutar la acción
        self.robot.step(1250)  # Usa el timestep correcto
        

    def puede_moverse_adelante(self):
        # Verificar si puede avanzar sin colisionar
        for sensor in self.sensores_delante:
            if sensor.getValue() > self.umbral:  # Observar el umbral
                return False
        return True  # Puede avanzar

    def puede_moverse_detras(self):
        # Verificar si puede retroceder sin colisionar
        for sensor in self.sensores_detras:
            if sensor.getValue() > self.umbral:  # Observar el umbral
                return False
        return True  # Puede retroceder
    
    def recompensa(self):
        # Calcular recompensa basada en la distancia al objetivo
        robot_pos = np.array(self.robot_translation_field.getSFVec3f())
        target_pos = np.array(self.target_translation_field.getSFVec3f())
        distance = np.linalg.norm(robot_pos - target_pos)

        if distance < 0.1:  # Cerca del objetivo
            return 100
        elif distance > 2.0:  # Muy lejos del objetivo
            return -10
        else:  # Dentro de una distancia aceptable
            return 0
    
    def reset(self):
        # Restablecer la simulación y la física
        self.supervisor.simulationResetPhysics()
        self.supervisor.simulationReset()

        # Reiniciar la velocidad de los motores
        self.motor_izq.setVelocity(0)
        self.motor_der.setVelocity(0)

        # Avanzar un paso para estabilidad
        self.robot.step(self.timestep)
