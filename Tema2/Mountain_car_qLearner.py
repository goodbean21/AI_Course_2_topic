# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:25:17 2019

@author: Usuario
"""

import gym
import numpy as np

"""
EPSILON_MIN       : Se aprende mientras el valor de Epsilon sea superior que Epsilon_MIN
MAX_NUM_EP        : Número máximo de iteraciones
STEPS_PER_EP      : Número de pasos por episodios
ALPHA             : Ratio de aprendizaje
GAMMA             : Descuento 
MIN_DISCRETE_BINS : Número mínimo de discretizaciones del espacio continuo
"""

MAX_NUM_EP = 10000
MAX_STEP_PER_EP = 200

EPSILON_MIN = 0.005
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

MAX_NUM_STEPS = MAX_NUM_EP * MAX_STEP_PER_EP
EPSILON_DECAY = 500 * EPSILON_MIN/MAX_NUM_STEPS

class QLearner(object):
    def __init__(self, env):
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0
        
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high - self.obs_low)/self.obs_bins
        
        self.action_shape = env.action_space.n
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))

    def get_Action(self, obs):
        discrete_obs = self.discretize(obs)
        # Selección de la acción en base a Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        
         # Probabilidad (1 - Epsilon)
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discrete_obs])
        
        # De otra forma con probabilidad Epsilon
        return np.random.choice([a for a in range(self.action_shape)])
        
    def discretize(self, obs):
        return tuple(((obs - self.obs_low)/self.obs_width).astype(int))
        
    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        
        self.Q[discrete_obs][action] += self.alpha*(reward + self.gamma * np.max(self.Q[discrete_next_obs]) - self.Q[discrete_obs][action])

# Método de entrenamiento 
def train(agent, env):
    best_reward = -float("inf")
    for episode in range(MAX_NUM_EP):
        done = False
        obs = env.reset()
        total_reward = 0.0
        
        while (not done):
            action = agent.get_Action(obs) # La action elegida por Q-Learner
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        print("Episodio número: {} con recompensa: {} y mejor recompensa: {}, epsilon: {}".format(episode, total_reward, best_reward, agent.epsilon))
        
    ## De todas las políticas de entrenamiento que hemos obtenido
    ## Devolmemos la mejor
    return np.argmax(agent.Q, axis = 2)

def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    
    while(not done):
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
        
    return total_reward

if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = QLearner(environment)
    learn_policy = train(agent, environment)
    monitor_path = "./monitor_out"
    env = gym.wrappers.Monitor(environment, monitor_path, force = True)
    for _ in range(10):
        test(agent, env, learn_policy)
        
    env.close()