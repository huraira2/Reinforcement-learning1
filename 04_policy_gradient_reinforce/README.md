# Policy Gradient REINFORCE

Implementation of the REINFORCE algorithm, a fundamental policy gradient method.

## Overview

Unlike value-based methods, this approach directly optimizes the policy using gradient ascent. The project shows how to compute policy gradients and update parameters to maximize expected rewards.

## Features

- REINFORCE algorithm implementation
- Policy network with PyTorch
- Baseline subtraction for variance reduction
- Monte Carlo policy gradient estimation
- Performance analysis and visualization

## Running the Project

```bash
cd 04_policy_gradient_reinforce
pip install torch gym
python policy_gradient_reinforce.py
```

## Key Concepts

- **Policy Gradients**: Direct policy optimization
- **REINFORCE**: Monte Carlo policy gradient
- **Baseline**: Variance reduction technique
- **Policy Networks**: Neural network policies

## Requirements

- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib