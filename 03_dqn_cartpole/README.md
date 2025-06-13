# Deep Q-Network (DQN) CartPole

Deep reinforcement learning implementation using Deep Q-Networks to solve the CartPole environment.

## Overview

This project combines neural networks with Q-learning, introducing concepts like experience replay and target networks. It demonstrates how deep learning can be applied to RL problems with continuous state spaces.

## Features

- Deep Q-Network implementation with PyTorch
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration with decay
- Performance visualization and monitoring

## Running the Project

```bash
cd 03_dqn_cartpole
pip install torch gym
python dqn_cartpole.py
```

## Key Concepts

- **Deep Q-Networks**: Neural networks for Q-value approximation
- **Experience Replay**: Learning from stored experiences
- **Target Networks**: Stable learning targets
- **Function Approximation**: Handling continuous state spaces

## Requirements

- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib