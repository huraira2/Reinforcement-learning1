# A3C (Asynchronous Advantage Actor-Critic)

Asynchronous Advantage Actor-Critic implementation demonstrating parallel learning across multiple environments.

## Overview

A3C uses multiple workers to collect experience asynchronously, leading to more diverse training data and faster convergence. This project explores advanced concepts in distributed reinforcement learning.

## Features

- A3C algorithm implementation
- Multiple parallel workers
- Asynchronous gradient updates
- Shared global network
- Performance comparison with synchronous methods

## Running the Project

```bash
cd 09_a3c_async
pip install torch gym
python a3c_async.py
```

## Key Concepts

- **Asynchronous Learning**: Parallel environment interaction
- **Shared Networks**: Global parameter sharing
- **Worker Threads**: Independent learning agents
- **Gradient Accumulation**: Asynchronous parameter updates

## Requirements

- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib
- Threading support