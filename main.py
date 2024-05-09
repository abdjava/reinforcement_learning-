import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep

animate = False
lr = 0.1  # learning rate
discount = 0.95  # discount factor
episode = 25000
display = 500
epsilon = 0.5  # chance we explore instead of trying optimal action
start_decay = 1  # which episode we start to decay
stop_decay = episode // 2
decay_amount = epsilon / (stop_decay - start_decay)

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.reset()
state_size = [20] * len(env.observation_space.high)
state_width = (env.observation_space.high - env.observation_space.low) / state_size

Q_table = np.random.uniform(low=-2, high=0, size=(state_size + [env.action_space.n]))

display_stack = []


def plot_ani(stack):
    fig_stack = []
    for img in stack:
        fig_stack.append([plt.imshow(img, animated=True)])

    return fig_stack


def get_discrete(state):
    state_discrete = (state - env.observation_space.low) / state_width
    return tuple(state_discrete.astype(int))


success_rate = 0
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
        new_s, reward, finished, compleat, _ = env.step(action)

        finished = compleat or finished
        new_state_d = get_discrete(new_s)

        # calcating next action and table
        if not finished:

            Max_future = np.max(Q_table[new_state_d])
            Q_value_new = (1 - lr) * Q_table[state_d + (action,)] + lr * (reward + discount * Max_future)
            Q_table[state_d + (action,)] = Q_value_new

        elif new_s[0] >= env.get_wrapper_attr('goal_position'):
            success_rate += 1
            Q_table[state_d + (action,)] = 0
        state_d = new_state_d

        if ep % display == 0 and animate:
            display_stack.append(env.render())

    if stop_decay > ep > start_decay and epsilon > 0:
        epsilon -= decay_amount
    elif epsilon < 0:
        epsilon = 0

    if ep % display == 0:
        print(f" Last {display} ({ep}/{episode}) success rate {success_rate / display}")
        success_rate = 0
        if animate:
            fig = plt.figure()
            frames = plot_ani(display_stack)
            ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=30)
            plt.show()
            sleep(0.1)
            display_stack = []

env.close()
print("program is finished ")
