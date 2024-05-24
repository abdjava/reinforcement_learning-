"""
This file contains the Learning Manager class that manages the reinforcement learning processing.
It has implementation for various algorithms used to estimate value of polices and update agent parameters

Author : Abdul Basit
"""
from agent import Agent
import numpy as np
import gymnasium as gym
from tqdm import tqdm


class LearningManager:
    """
    This is the Learning Manager class contain algorithms for monte carlo estimation of a policy given an agent
    """

    def __init__(self, env: gym.Env, ep: int, discount_factor: float = 1):
        self.ep = ep
        self.discount_factor = discount_factor
        self.env = env
        self.state_space_sizes = [s.n for s in env.observation_space]

    def monte_carlo_estimation(self, agent: Agent) -> tuple:
        """
        This method runs a monte carlo estimation of the value of the agent's policy. This is done using a finite state and action space.
        :param agent:[Agent]  The agent to evaluate
        :return:value and visits: [ndarray,ndarray] Returns a table of state values and number of visits with index by the state space
        """
        value_table = np.zeros(self.state_space_sizes, dtype=float)
        num_vists_table = np.zeros(self.state_space_sizes, dtype=float)

        for i in tqdm(range(self.ep)):
            finished = False
            if_visted = np.zeros(self.state_space_sizes, dtype=bool)  # if visited this ep
            histroy = []
            state = self.env.reset()[0]
            histroy.append((state, 0, 0))

            # generate episode
            while not finished:
                # pick action
                action = agent.get_action(state)
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                finished = terminated or truncated
                histroy.append((new_state, reward, int(finished)))
                state = new_state

            # update value function with first vist
            for j, (s, r, f) in enumerate(histroy):
                if not if_visted[s]:
                    reward = self.calculate_return(histroy[j + 1:]) if len(histroy[j + 1:]) else r
                    if_visted[s] = True
                    num_vists_table[s] += 1
                    value_table[s] = (value_table[s] * (num_vists_table[s] - 1) + reward) / num_vists_table[s]

        return value_table, num_vists_table

    def calculate_return(self, histroy) -> float:
        if not len(histroy):
            return -1

        discounted_sum = 0
        for i, (s, r, f) in enumerate(histroy):
            discounted_sum = discounted_sum + self.discount_factor ** i * r
        return discounted_sum
