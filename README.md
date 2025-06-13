# Reinforcement Learning Projects Collection

This repository contains ten comprehensive reinforcement learning projects, each demonstrating different algorithms, environments, and applications. From basic Q-learning to advanced deep reinforcement learning techniques, these projects provide hands-on experience with the fundamental concepts of RL.

## Projects Overview

### 1. Q-Learning Grid World
**Location:** `01_q_learning_gridworld/`

A classic implementation of Q-learning algorithm in a simple grid world environment. The agent learns to navigate from a start position to a goal while avoiding obstacles. This project demonstrates the fundamentals of temporal difference learning and the Q-learning update rule. Perfect for understanding how agents learn optimal policies through exploration and exploitation.

### 2. Multi-Armed Bandit
**Location:** `02_multi_armed_bandit/`

Implementation of various strategies for the multi-armed bandit problem, including epsilon-greedy, UCB (Upper Confidence Bound), and Thompson sampling. This project explores the exploration vs exploitation dilemma, which is fundamental to reinforcement learning. Learn how different algorithms balance trying new actions versus exploiting known good actions.

### 3. Deep Q-Network (DQN) CartPole
**Location:** `03_dqn_cartpole/`

A deep reinforcement learning implementation using Deep Q-Networks to solve the CartPole environment. This project combines neural networks with Q-learning, introducing concepts like experience replay and target networks. Demonstrates how deep learning can be applied to RL problems with continuous state spaces.

### 4. Policy Gradient REINFORCE
**Location:** `04_policy_gradient_reinforce/`

Implementation of the REINFORCE algorithm, a fundamental policy gradient method. Unlike value-based methods, this approach directly optimizes the policy using gradient ascent. The project shows how to compute policy gradients and update parameters to maximize expected rewards, providing insight into policy-based reinforcement learning.

### 5. Actor-Critic Mountain Car
**Location:** `05_actor_critic_mountain_car/`

An actor-critic implementation solving the Mountain Car problem. This project combines the benefits of both value-based and policy-based methods, where the actor learns the policy and the critic evaluates it. Demonstrates how actor-critic methods can provide more stable learning compared to pure policy gradient approaches.

### 6. SARSA Taxi Environment
**Location:** `06_sarsa_taxi/`

Implementation of the SARSA (State-Action-Reward-State-Action) algorithm in the Taxi environment. Unlike Q-learning, SARSA is an on-policy method that learns the value of the policy being followed. This project highlights the differences between on-policy and off-policy learning methods.

### 7. Double DQN Lunar Lander
**Location:** `07_double_dqn_lunar_lander/`

Advanced DQN implementation with Double DQN improvements to solve the Lunar Lander environment. This project addresses the overestimation bias in standard DQN by using two networks for action selection and evaluation. Demonstrates how algorithmic improvements can significantly enhance learning performance.

### 8. PPO (Proximal Policy Optimization) Continuous Control
**Location:** `08_ppo_continuous_control/`

Implementation of Proximal Policy Optimization for continuous control tasks. PPO is a state-of-the-art policy gradient method that maintains a balance between sample efficiency and implementation simplicity. This project shows how to handle continuous action spaces and implement clipped surrogate objectives.

### 9. A3C (Asynchronous Advantage Actor-Critic)
**Location:** `09_a3c_async/`

Asynchronous Advantage Actor-Critic implementation demonstrating parallel learning across multiple environments. A3C uses multiple workers to collect experience asynchronously, leading to more diverse training data and faster convergence. This project explores advanced concepts in distributed reinforcement learning.

### 10. Multi-Agent Reinforcement Learning
**Location:** `10_multi_agent_rl/`

A multi-agent reinforcement learning environment where multiple agents learn to cooperate or compete. This project introduces concepts like independent learning, centralized training with decentralized execution, and communication between agents. Demonstrates how RL extends to complex multi-agent scenarios.

## Getting Started

Each project is self-contained with its own requirements and instructions. Navigate to any project directory and follow the README for specific setup and running instructions.

### General Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Gym/Gymnasium (for OpenAI environments)
- PyTorch or TensorFlow (for deep learning projects)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Reinforcement-learning1

# Install common dependencies
pip install numpy matplotlib gym torch torchvision
```

## Learning Path

For beginners, we recommend following the projects in order:
1. Start with **Q-Learning Grid World** to understand basic RL concepts
2. Explore **Multi-Armed Bandit** for exploration strategies
3. Move to **DQN CartPole** for deep RL introduction
4. Continue with **Policy Gradient** and **Actor-Critic** methods
5. Advance to more sophisticated algorithms like **PPO** and **A3C**
6. Conclude with **Multi-Agent RL** for complex scenarios

## Contributing

Feel free to contribute improvements, bug fixes, or additional projects. Please ensure code follows the existing style and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.