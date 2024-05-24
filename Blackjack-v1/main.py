import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agent import Agent, PickUnder20
from learningmanager import LearningManager

eps = 500000
discount = 0.95
env = gym.make('Blackjack-v1', natural=False, sab=False)

# monte carlo estimation

agent = PickUnder20()
LM = LearningManager(env, eps, discount)

value_table, num_vists_table = LM.monte_carlo_actionvalue_estimation(agent)
print(value_table[21, 2, 1])
print(num_vists_table[21, 2, 1])

print(value_table[4, 10, 0])
print(num_vists_table[4, 10, 0])
