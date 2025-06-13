# Multi-Armed Bandit

Implementation of various strategies for the multi-armed bandit problem.

## Overview

This project explores the exploration vs exploitation dilemma through different bandit algorithms including epsilon-greedy, UCB (Upper Confidence Bound), and Thompson sampling.

## Features

- Multiple bandit algorithms implementation
- Comparative analysis of different strategies
- Visualization of performance over time
- Configurable number of arms and reward distributions

## Running the Project

```bash
cd 02_multi_armed_bandit
python multi_armed_bandit.py
```

## Algorithms Implemented

- **Epsilon-Greedy**: Simple exploration strategy
- **UCB (Upper Confidence Bound)**: Optimistic exploration
- **Thompson Sampling**: Bayesian approach to exploration
- **Greedy**: Pure exploitation baseline

## Key Concepts

- **Exploration vs Exploitation**: Fundamental RL trade-off
- **Regret**: Cumulative difference from optimal performance
- **Confidence Bounds**: Statistical approach to uncertainty