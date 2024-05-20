"""
This file contains several polices for the Blackjack-v1 game

author: Abdul Basit
"""
from typing import Tuple

import gymnasium as gym


def pick_police(state: Tuple[int, int, int]) -> int:
    """
    This policy picks new card if value under 20 else does a stick
    """
    return 1 if state[0] < 20 else 0
