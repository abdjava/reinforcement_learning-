# Mountain Car v0

This directory solves the [mountain car with **discrete actions**](https://gymnasium.farama.org/environments/classic_control/mountain_car/). This is
done using by discretizing the state and using a look-up table to represent a Q function.
Then Q learning is used update the table to learn the optimal policy.