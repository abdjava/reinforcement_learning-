import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

state_size = [20] * len(env.observation_space.high)
state_width = (env.observation_space.high - env.observation_space.low) / state_size
print(state_width)

Q_table = np.random.uniform(low=-2, high=0, size=(state_size + [env.action_space.n]))
print(Q_table.shape)
finished = False

while not finished:
    action = 2
    new_s, reward, finished, _ = env.step(action)
    env.render()

env.close()
print("program is finished ")
