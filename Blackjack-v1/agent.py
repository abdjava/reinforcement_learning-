"""
This file contains various different agents for machine learning all delivered from the based agent class

Author: Abdul Basit
"""
from abc import ABC
from typing import Optional as Op, Tuple
import abc
import numpy as np


class Agent(abc.ABC):
    """
    This is the base class for all agents. Any new agent should inherent from this class
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, args) -> Op[int]:
        """
        Given inputs should always return an action indexed by ints
        :return: Action [int]
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> Op[np.ndarray]:
        """
        Return the internal parameters of the class as ndarray
        :return: Parameters [ndarray]
        """
        pass


class PickUnder20(Agent):
    """
    Simple policy that hits if score under 20 else passes
    """

    def __init__(self):
        pass

    def get_action(self, state: Tuple[int, int, int]) -> int:
        """
        Hit if state[0] < 20 else pass
        :return: action [1 or 0]
        """
        return 1 if state[0] < 20 else 0

    def get_parameters(self) -> Op[np.ndarray]:
        """
        This policy has no parameters
        :return: None
        """
        return None
