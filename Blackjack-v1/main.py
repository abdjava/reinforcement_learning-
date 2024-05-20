import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
from policies import pick_police
from tqdm import tqdm


def calculate_return(histroy, lamda=1) -> float:
    if not len(histroy):
        return -1

    discounted_sum = 0
    for i, (s, r, f) in enumerate(histroy):
        discounted_sum = discounted_sum + lamda ** i * r
    return discounted_sum


eps = 500000
env = gym.make('Blackjack-v1', natural=False, sab=False)

# monte carlo estimation

state_space_sizes = [s.n for s in env.observation_space]
value_table = np.zeros(state_space_sizes, dtype=float)
num_vists_table = np.zeros(state_space_sizes, dtype=float)

for i in tqdm(range(eps)):
    finished = False
    if_visted = np.zeros(state_space_sizes, dtype=bool)  # if visited this ep
    histroy = []
    state = env.reset()[0]
    histroy.append((state, 0, 0))
    # generate episode
    while not finished:
        # pick action
        action = pick_police(state)
        new_state, reward, terminated, truncated, _ = env.step(action)

        finished = terminated or truncated
        histroy.append((new_state, reward, int(finished)))
        state = new_state

    # update value function with first vist
    for i, (s, r, f) in enumerate(histroy):
        if not if_visted[s]:
            reward = calculate_return(histroy[i + 1:]) if len(histroy[i + 1:]) else r
            if_visted[s] = True
            num_vists_table[s] += 1
            value_table[s] = (value_table[s] * (num_vists_table[s] - 1) + reward) / num_vists_table[s]

print(value_table[21, 2, 1])
print(num_vists_table[21, 2, 1])

print(value_table[4, 10, 0])
print(num_vists_table[4, 10, 0])
