# CartPole-v1
This project demonstrates a soloution to the CartPole-v1 problem of the gymnasium library. You can see the information about this problem [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
The problem is solved using SARSA and DQN algorithms based on a neural network structure for state-value function, with two different policies ( epsilon-greedy and Boltzmann ), The algorithms along with their policies and NN structures are completely implemented in the [Main.py](Main.py) file.
## Requirements
This code uses pytorch and gymnasium for training the agent and interacting with the environment, so make sure you install these libraries correctly. Other libraries include:
```python
import os
import gc
import pygame
import warnings
import numpy as np
import gymnasium as gym
from collections import deque
import math
import matplotlib.pyplot as plt
```
## Reward Wrapper
Before showing the results of each algorithm lets show a reward wrapper which I've written for better guidance for the agent. It uses cosin function for reward calculation. I did this because I wanted the agent to stay at the center position with angle of 0 radians. So the wrapper gives maximum reward at the center, Then, the more the agent moves away from the stable state the less the reward would be. And, the more the agent gets closer to the boundry points of the game (which the game finishes if the agents passes these boundry points), it gets more and more negative rewards. Finaly, the gain of agent's angle reward is more than the agent's position reward, Because, I wanted the agent to pay more attention to maintaining the pole in more stable angle rather than being in the center position.
```python
    def reward(self, state):
        """
        Modifies the reward based on the current state of the environment.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            float: The modified reward.
        """
        current_position = state[0]
        current_angel = state[2]

        position_reward = math.cos((2 * math.pi * current_position) / 4.8)
        angel_reward = math.cos((2 * math.pi * current_angel) / 0.419)

        modified_reward = position_reward + 2 * angel_reward
        return modified_reward
```
## Epsilon-greedy and Boltzmann policies
Here, is the implementation for epsilon-greedy and Boltzmann policiec:
```python
    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy OR Boltzmann strategy(specified by  self.epsilon_or_boltzmann).

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """
        if self.epsilon_or_boltzmann:
            # Exploration: epsilon-greedy
            if np.random.random() < self.epsilon_max:
                return self.action_space.sample()

            # Exploitation: the action is selected based on the Q-values.
            # Check if the state is a tensor or not. If not, make it a tensor
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                Q_values = self.main_network(state)
                action = torch.argmax(Q_values).item()
        else:  # Exploration: Boltzmann.
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                q = self.main_network(state)
                q /= self.temp  # dividing each Q(s, a) by the temperature.
                q = torch.nn.functional.softmax(q, dim=0)  # calculating softmax of each Q(s, a)/temp.
                # now, sampling an action using the multinomial distribution calculated above:
                action = torch.multinomial(q, 1).item()

        return action
```
## Results
I've trained the model for DQN algorithm with both e-greedy and Boltzmann policies, and SARSA just with e-greedy policy.
## DQN Epsilon-greedy
Steps played: 500 (maximum steps, which means the agent kept the pole in a steady state for 500 steps and finished the game successfully.)
Raw rewared gained: 1475.94


![Agent's reward chart](/images/dqn_epsilon_reward.jpg)
![Agent's loss chart](/images/dqn_epsilon_loss.jpg)
![Agent's game play](/images/dqn_epsilon.gif)
## DQN Boltzmann
Steps played: 500 (maximum steps, which means the agent kept the pole in a steady state for 500 steps and finished the game successfully.)
Raw rewared gained: 1362.23


![Agent's reward chart](/images/dqn_boltzmann_reward.jpg)
![Agent's loss chart](/images/dqn_boltzmann_loss.jpg)
![Agent's game play](/images/dqn_boltzmann.gif)
## SARSA Epsilon-greedy
Steps played: 500 (maximum steps, which means the agent kept the pole in a steady state for 500 steps and finished the game successfully.)
Raw rewared gained: 1484.33


![Agent's reward chart](/images/sarsa_reward.jpg)
![Agent's loss chart](/images/sarsa_loss.jpg)
![Agent's game play](/images/sarsa.gif)

## Thanks to
Thanks to [Mehdi Shabazi](https://github.com/MehdiShahbazi) who provided us with a base code and guided us how to implement the NN structure and DQN agent.
