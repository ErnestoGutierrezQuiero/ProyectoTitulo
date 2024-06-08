from controller import Robot, Supervisor
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from random import sample, choice


start_acc = 0.5
trainer_num = 2
base_rate = 0.5
feedback_prob = 0

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
    
    def dentro_de_limites(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
   
#IMPLEMENTACION MTIRL   ------------------------------------------------------
 
    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    
    def get_state(self):
        e = self.estado
        if not isinstance(e, (tuple, list)):
            raise ValueError("Estado must be a tuple or list")
        return tuple(int(x) for x in e)
    
    def _init_state_value(self, s_name, randomized = False):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.action_space.n):
                default_v = random.random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
                
    def _assert_state_in_Q(self, s, randomized = False):
        # 　cann't find the state
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)
            
    def performPolicy(self, estado, n_episodio, use_epsilon=True):
        return self._curPolicy(estado,n_episodio,use_epsilon)
    
    def curPolicy(self, estado, n_episodio, use_epsilon):
        epsilon=1.00/n_episodio
        Q_s = self.Q[estado[0],estado[1]]
        if use_epsilon and random.random()<epsilon:
            action = np.random.randint(6)
        else:
            action = np.argmax(Q_s)
        return action
    
    def _curPolicy(self, estado, n_episodio, use_epsilon=True):
    # Debugging: Print estado
        print(f"Debug: estado = {estado}")
    # Check if estado is a tuple or list of integers
        if not (isinstance(estado, (tuple, list)) and all(isinstance(x, int) for x in estado)):
            raise ValueError(f"Invalid estado: {estado}. Expected a tuple or list of integers.")
    # Check if the indices are within the range of Q
        if estado[0] < 0 or estado[0] >= self.Q.shape[0] or estado[1] < 0 or estado[1] >= self.Q.shape[1]:
            raise IndexError(f"Index out of range: {estado}. Q shape: {self.Q.shape}")
        Q_s = self.Q[estado[0], estado[1]]
        epsilon = 1.0 / (1 + n_episodio)  # Ensure epsilon is defined
    # Select action based on epsilon-greedy policy
        if use_epsilon and random.random() < epsilon:
            action = choice(range(len(Q_s)))
        else:
            action = np.argmax(Q_s)
        return action

    
    def Q_get(self):
        return self
    def _is_state_in_Q(self, s):
        return s in self.Q

    def _get_state_name(self, state):
        return tuple(state)

    def _get_Q(self, s, a):
        return self.Q[s[0], s[1], a]

    def _set_Q(self, s, a, value):
        self.Q[s[0], s[1], a] = value
    
    def entrenamientoMTIRL(self, gamma, alpha, max_num_episodes):
        total_steps=0
        num_episode=1
        step_list=[]
        action_max=  np.zeros((self.num_boxes_x, self.num_boxes_y, 6))  # 6 acciones posibles
        state_fedbk_time=0
        state_ac_mem={}

        state_first_acc=0
        state_final_set={}
        while num_episode<=max_num_episodes:
            self.reset()
            s0 = self.get_state()
            a0 = self.performPolicy(s0, num_episode,use_epsilon=True)
            step_in_episode = 0
            is_done = False
            r_num1=0
            prob_cor=0
            prob_wro=0
            while not is_done:
                s0=tuple(s0)
                s1, r1 , is_done = self.step(a0)
                r_num = r1
                a1=self.performPolicy(s1,num_episode,use_epsilon=False)
                old_q=self._get_Q(s0,a0)
                
                with open("q_table.pkl", 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                dict_data = {}
                for key, value in dict:
                    dict_data[key] = value
                print("---")
                action_o = dict_data[tuple(s0)]
                action_o1 = sorted(action_o.items(), key=lambda kv: (kv[1], kv[0]))
                action_o2 = list(zip(*action_o1))[0]
                print(action_o)
                print(action_o1)
                print(action_o2)
                        
                results=[]
                r1=0
                if (s0,a0) in state_ac_mem.keys():
                    fb_pro = 1-state_ac_mem[s0,a0]
                else:
                    fb_pro =1
                if fb_pro >random.random():
                    state_fedbk_time+=1
                    for j in range(trainer_num):
                        action_judge = [-1,-1,-1,-1,-1,0]
                        if action_o1[3][1] == action_o1[2][1]:
                            action_judge[2] = action_judge[3]
                        feedback_prob_every = random.random()
                        if feedback_prob_every > feedback_prob:
                            rand_t1 = random.random()
                            if rand_t1 < trainer_acc[j]:
                                if a0 == action_o2[0]:
                                    q_extend = action_judge[0]
                                elif a0 == action_o2[1]:
                                    q_extend = action_judge[1]
                                elif a0 == action_o2[2]:
                                    q_extend = action_judge[2]
                                elif a0 == action_o2[3]:
                                    q_extend = action_judge[3]
                                elif a0 == action_o2[4]:
                                    q_extend = action_judge[4]
                                elif a0 == action_o2[5]:
                                    q_extend = action_judge[5]
                                re_t1  = q_extend

                            else:
                                if a0 == action_o2[0]:
                                    action_judge.remove(action_judge[0])
                                    q_extend = sample(action_judge, 1)[0]
                                elif a0 == action_o2[1]:
                                    action_judge.remove(action_judge[1])
                                    q_extend = sample(action_judge, 1)[0]
                                elif a0 == action_o2[2]:
                                    action_judge.remove(action_judge[2])
                                    q_extend = sample(action_judge, 1)[0]
                                elif a0 == action_o2[3]:
                                    action_judge.remove(action_judge[3])
                                    q_extend = sample(action_judge, 1)[0]
                                elif a0 == action_o2[4]:
                                    action_judge.remove(action_judge[4])
                                    q_extend = sample(action_judge, 1)[0]
                                elif a0 == action_o2[5]:
                                    action_judge.remove(action_judge[5])
                                    q_extend = sample(action_judge, 1)[0]                                
                                re_t1  = q_extend

                        else:
                            re_t1 = 2
                        # results.append([j,re_t1])
                        # Escribe el primer grupo de información de retroalimentación histórica.
                        if re_t1 !=2:
                            trainer_mem[j][s0,a0] = re_t1
                # Calcular el resultado actual
                for tm in range(len(trainer)):
                    if (s0,a0) in trainer_mem[tm].keys():
                        #Agregar resultados a results
                        results.append([tm,trainer_mem[tm][s0,a0]])
                # print(trainer_mem)
                y_prob = 1
                n_prob = 1
                y_set = []
                n_set = []
                # print(results)
                for k in range(len(results)):
                    if results[k][1] == 0:
                        y_set.append(results[k][0])
                    if results[k][1] == -1:
                        n_set.append(results[k][0])

                if y_set !=[] or n_set!=[]:
                    fb_a = 1
                    aver_un = 0
                    if len(y_set) >0 or len(n_set)>0:
                        for m in y_set:
                            aver_un += 2/(trainer[m][2]+trainer[m][3]+2)
                        for n in n_set:
                            aver_un += 2/(trainer[n][2]+ trainer[n][3]+2)
                        aver_un =aver_un/(len(y_set)+len(n_set))

                    y_prob = 1
                    n_prob = 1
                    y_prob_v = 0
                    n_prob_v = 0
                    for m in y_set:
                        y_prob_v += trainer[m][1]
                    for n in n_set:
                        n_prob_v += trainer[n][1]
                    y_prob_v = round(y_prob_v,25)
                    n_prob_v = round(n_prob_v,25)
                    for m in y_set:
                        y_prob *= (trainer[m][1])

                        n_prob *= (1-(trainer[m][1]))
                    for n in n_set:
                        y_prob *= (1-(trainer[n][1]))
                        n_prob *= (trainer[n][1])

                    y_prob = round(y_prob,25)
                    n_prob = round(n_prob,25)
                    cor_prob_y = y_prob/(y_prob+n_prob)
                    cor_prob_n = n_prob/(y_prob+n_prob)

                    cor_probv_y = y_prob_v/(y_prob_v+n_prob_v)
                    cor_probv_n = n_prob_v/(y_prob_v+n_prob_v)

                    y_prob_mix = (1-aver_un)*cor_prob_y + aver_un* cor_probv_y
                    n_prob_mix = (1-aver_un)*cor_prob_n + aver_un* cor_probv_n

                    if y_prob_mix > n_prob_mix:
                        r1 = 0
                        #Si eliges la acción correcta
                        if a0 == action_o2[3] or(a0 == action_o2[2] and action_o1[3][1] == action_o1[2][1]):

                            state_final_set[s0,a0] = 1
                            if (s0,a0) not in state_ac_mem.keys():
                                state_first_acc += 1
                        else:
                            state_final_set[s0,a0] = 0

                        cor_prob = y_prob_mix/(y_prob_mix+n_prob_mix)
                        update_value = cor_prob-(1-cor_prob)
                        state_ac_mem[s0,a0] = update_value

                        # Solo agregue los valores de estas personas si hay comentarios.
                        if fb_a == 1:
                            # for m in set1_up:
                            for m in y_set:
                                trainer[m][2] += update_value
                                # a[m][2] += cor_prob *(pow((np.log(len(y_set)+len(n_set)))/(np.log(set_num+set_num)),1/100))
                                trainer[m][1] = (trainer[m][2]/(trainer[m][2]+trainer[m][3]+2)+
                                                 0.75*(2/(trainer[m][2]+trainer[m][3]+2)))
                            # for n in set2_up:
                            for n in n_set:
                                trainer[n][3] += update_value
                                # a[n][3] += cor_prob *(pow((np.log(len(y_set)+len(n_set)))/(np.log(set_num+set_num)),1/2))
                                trainer[n][1] = (trainer[n][2]/(trainer[n][2]+trainer[n][3]+2)+
                                                 0.75*(2/(trainer[n][2]+trainer[n][3]+2)))

                    if y_prob_mix < n_prob_mix:
                        r1 = -1
                        if a0 != action_o2[3] or (a0 ==action_o2[2] and action_o1[3][1] != action_o1[2][1]):
                            prob_cor +=1
                            state_final_set[s0,a0] = 1
                            if (s0,a0) not in state_ac_mem.keys():
                                state_first_acc += 1

                        else:
                            state_final_set[s0,a0] = 0
                            prob_wro +=1
                        cor_prob = n_prob_mix/(y_prob_mix+n_prob_mix)
                        update_value = cor_prob-(1-cor_prob)
                        if (s0,a0) in state_ac_mem.keys():
                            kuppra = -state_ac_mem[s0,a0]+ update_value
                        else:
                            kuppra = update_value
                        state_ac_mem[s0,a0] = update_value
                        if fb_a == 1:
                            # for m in set1_up:
                            for m in y_set:
                                trainer[m][3] += update_value
                                # a[m][3] += cor_prob *(pow((np.log(len(y_set)+len(n_set)))/(np.log(set_num+set_num)),1/2))
                                trainer[m][1] = (trainer[m][2]/(trainer[m][2]+trainer[m][3]+2)+
                                                 base_rate*(2/(trainer[m][2]+trainer[m][3]+2)))
                            # for n in set2_up:
                            for n in n_set:
                                trainer[n][2] += update_value
                                #         # a[n][2] += cor_prob*(pow((np.log(len(y_set)+len(n_set)))/(np.log(set_num+set_num)),1/2))
                                trainer[n][1] = (trainer[n][2]/(trainer[n][2]+trainer[n][3]+2)+
                                                 base_rate*(2/(trainer[n][2]+trainer[n][3]+2)))

                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)
#---
                # Comparar la bestaction convergente
                with open("best_actions.pkl", 'rb') as fo:     
                    action_total = pickle.load(fo, encoding='bytes')

                # Create the current table of best actions
                action_max[s0] = int(np.argmax(self.Q[s0]))

                # Compare the same number of actions as the optimal strategy
                action_number = 0
                diff = 0
                for i in range(len(action_total)):
                    if np.array_equal(action_max[i], action_total[i]):
                        action_number += 1
                    else:
                        diff += 1

#-----
                # Recompensas acumulativas
                r_num1 +=r_num

                s0, a0 = s1, a1
                step_in_episode += 1
                if step_in_episode >100:
                    break
                

            step_list.append(step_in_episode)
            total_steps += step_in_episode
            num_episode += 1

        state_final_acc = 0
        for fa in state_final_set.keys():
            if state_final_set[fa] == 1:
                state_final_acc+=1
        febset = []
        febset.append(state_fedbk_time)
        febset.append(len(state_ac_mem))
        febset.append(state_first_acc)
        febset.append(state_final_acc)
        print(febset)
        # self.experience.last.print_detail()
        return action_number, step_list, febset
                    
def main():
    print("esto sigue(2)")
    agent=MiRobotSupervisor()
    print("Learning...")
    conv_every,step_every,fedbak_set_every = agent.entrenamientoMTIRL(gamma=0.9,alpha=0.1,max_num_episodes=500)
    return conv_every,step_every,fedbak_set_every


start_acc=0.5
num_entrenadores=2
if __name__=='__main__':
    for sigma in [0.2]:
            action = []
            fedbak_set= []
            step = []
            for mu_row in range(51):
                mu = mu_row/100+0.5
                action_num = []
                febset_num = []
                step_num = []
                for i in range(100):
                    print(mu,sigma,i)
                    trainer_acc = np.random.normal(mu, sigma, num_entrenadores)
                    trainer = []
                    for trainer_n in range(num_entrenadores):
                        if trainer_acc[trainer_n] >1:
                            trainer_acc[trainer_n] = 1
                        elif trainer_acc[trainer_n] <0:
                            trainer_acc[trainer_n] = 0
                        trainer.append([trainer_n, start_acc,0,0,0])
                    trainer_mem = [{} for ii in range(num_entrenadores)]
                    action_every, step_every,febset_every = main()
                    action_num.append(action_every)
                    febset_num.append(febset_every)
                    step_num.append(step_every)
                    print("action_num",action_num)

                action.append(action_num)
                fedbak_set.append(febset_num)
                step.append(step_num)
