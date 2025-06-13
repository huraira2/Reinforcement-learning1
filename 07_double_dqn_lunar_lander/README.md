# Double DQN Lunar Lander

Advanced DQN implementation with Double DQN improvements to solve the Lunar Lander environment.

## Overview

This project addresses the overestimation bias in standard DQN by using two networks for action selection and evaluation. Demonstrates how algorithmic improvements can significantly enhance learning performance.

## Features

- Double DQN implementation
- Lunar Lander environment
- Experience replay with prioritized sampling
- Target network updates
- Performance analysis and comparison with standard DQN

## Running the Project

```bash
cd 07_double_dqn_lunar_lander
pip install torch gym box2d-py
python double_dqn_lunar_lander.py
```

## Key Concepts

- **Double DQN**: Addressing overestimation bias
- **Action Selection vs Evaluation**: Decoupled Q-value estimation
- **Target Networks**: Stable learning targets
- **Continuous State Spaces**: Complex environment handling

## Requirements

- PyTorch
- OpenAI Gym
- Box2D (for Lunar Lander)
- NumPy
- Matplotlib