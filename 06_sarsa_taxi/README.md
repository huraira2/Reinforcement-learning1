# SARSA Taxi Environment

Implementation of the SARSA (State-Action-Reward-State-Action) algorithm in the Taxi environment.

## Overview

Unlike Q-learning, SARSA is an on-policy method that learns the value of the policy being followed. This project highlights the differences between on-policy and off-policy learning methods.

## Features

- SARSA algorithm implementation
- Taxi environment navigation
- On-policy vs off-policy comparison
- Policy visualization and analysis
- Performance metrics and learning curves

## Running the Project

```bash
cd 06_sarsa_taxi
python sarsa_taxi.py
```

## Key Concepts

- **SARSA**: On-policy temporal difference learning
- **On-policy vs Off-policy**: Different learning paradigms
- **Epsilon-greedy**: Exploration strategy
- **Taxi Problem**: Classic discrete RL environment

## Requirements

- OpenAI Gym
- NumPy
- Matplotlib