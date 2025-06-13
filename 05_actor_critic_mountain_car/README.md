# Actor-Critic Mountain Car

An actor-critic implementation solving the Mountain Car problem.

## Overview

This project combines the benefits of both value-based and policy-based methods, where the actor learns the policy and the critic evaluates it. Demonstrates how actor-critic methods can provide more stable learning compared to pure policy gradient approaches.

## Features

- Actor-Critic algorithm implementation
- Separate actor and critic networks
- Advantage estimation for policy updates
- Mountain Car environment solution
- Performance analysis and visualization

## Running the Project

```bash
cd 05_actor_critic_mountain_car
pip install torch gym
python actor_critic_mountain_car.py
```

## Key Concepts

- **Actor-Critic**: Combination of policy and value methods
- **Advantage Function**: Improved policy gradient estimation
- **Temporal Difference**: Critic learning with TD errors
- **Policy Improvement**: Actor updates using critic feedback

## Requirements

- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib