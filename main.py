import gymnasium as gym
import numpy as np


env = gym.make("MountainCar-v0")
env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)
# parameters

lr = 0.1
discount = 0.95
episode = 25000
display = 2000
epsilon = 0.5  # chance we explore instead of trying optimal action
start_decay = 1  # which episode we start to decay
stop_decay = episode // 2
decay_amount = epsilon / (stop_decay - start_decay)

state_size = [20] * len(env.observation_space.high)
state_width = (env.observation_space.high - env.observation_space.low) / state_size
# print(state_width)

Q_table = np.random.uniform(low=-2, high=0, size=(state_size + [env.action_space.n]))


# print(Q_table.shape)


def get_discrete(state):
    state_discrete = (state - env.observation_space.low) / state_width
    return tuple(state_discrete.astype(int))


for ep in range(episode):
    finished = False
    # reseting enviroment and geting argmax of first q value
    state_d = get_discrete(env.reset()[0])

    while not finished:
        if np.random.random() > epsilon:
            action = np.argmax(Q_table[state_d])
        else:
            action = np.random.randint(low=0, high=env.action_space.n)
        # doing the action
        new_s, reward, finished, compleat,_ = env.step(action)

        finished = compleat or finished
        new_state_d = get_discrete(new_s)

        # calcating next action and table
        if not finished:

            Max_future = np.max(Q_table[new_state_d])
            Q_value_new = (1 - lr) * Q_table[state_d + (action,)] + lr * (reward + discount * Max_future)
            Q_table[state_d + (action,)] = Q_value_new

        elif new_s[0] >= env.goal_position:
            print("won the game at ep = {}".format(ep))
            Q_table[state_d + (action,)] = 0
        state_d = new_state_d

        if ep % display == 0:
            print("current ep = {}".format(ep))
            env.render()

    if stop_decay > ep > start_decay and epsilon > 0:
        epsilon -= decay_amount
    elif epsilon < 0:
        epsilon = 0

env.close()
print("program is finished ")
