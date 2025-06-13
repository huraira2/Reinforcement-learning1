import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width-1, height-1)
        self.obstacles = [(2, 2), (1, 3), (3, 1)]
        self.reset()
    
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        new_pos = (
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        )
        
        # Check boundaries and obstacles
        if (0 <= new_pos[0] < self.height and 
            0 <= new_pos[1] < self.width and 
            new_pos not in self.obstacles):
            self.agent_pos = new_pos
        
        # Calculate reward
        if self.agent_pos == self.goal:
            reward = 100
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -100
            done = False
        else:
            reward = -1  # Small penalty for each step
            done = False
        
        return self.agent_pos, reward, done
    
    def get_valid_actions(self, state):
        return [0, 1, 2, 3]  # All actions always available
    
    def render(self, q_table=None):
        grid = np.zeros((self.height, self.width))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goal
        grid[self.goal] = 2
        
        # Mark agent
        grid[self.agent_pos] = 1
        
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, cmap='viridis')
        plt.colorbar(label='Cell Type')
        plt.title('Grid World Environment')
        
        # Add text annotations
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.start:
                    plt.text(j, i, 'S', ha='center', va='center', color='white', fontsize=12)
                elif (i, j) == self.goal:
                    plt.text(j, i, 'G', ha='center', va='center', color='white', fontsize=12)
                elif (i, j) in self.obstacles:
                    plt.text(j, i, 'X', ha='center', va='center', color='white', fontsize=12)
        
        plt.show()

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, valid_next_actions):
        current_q = self.q_table[state][action]
        
        if valid_next_actions:
            max_next_q = max([self.q_table[next_state][a] for a in valid_next_actions])
        else:
            max_next_q = 0
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_policy(self, state, valid_actions):
        q_values = [self.q_table[state][action] for action in valid_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

def train_agent(episodes=1000):
    env = GridWorld()
    agent = QLearningAgent()
    
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            valid_actions = env.get_valid_actions(state)
            action = agent.get_action(state, valid_actions)
            
            next_state, reward, done = env.step(action)
            valid_next_actions = env.get_valid_actions(next_state)
            
            agent.update_q_value(state, action, reward, next_state, valid_next_actions)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards_per_episode.append(total_reward)
        
        # Decay epsilon
        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.95)
            print(f"Episode {episode}, Average Reward: {np.mean(rewards_per_episode[-100:]):.2f}")
    
    return agent, rewards_per_episode

def visualize_policy(agent, env):
    """Visualize the learned policy"""
    policy_grid = np.zeros((env.height, env.width))
    action_symbols = ['↑', '→', '↓', '←']
    
    plt.figure(figsize=(10, 8))
    
    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state not in env.obstacles and state != env.goal:
                valid_actions = env.get_valid_actions(state)
                best_action = agent.get_policy(state, valid_actions)
                policy_grid[i, j] = best_action
                
                plt.text(j, i, action_symbols[best_action], 
                        ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Color the grid
    grid_colors = np.ones((env.height, env.width)) * 0.5
    for obs in env.obstacles:
        grid_colors[obs] = 0  # Black for obstacles
    grid_colors[env.goal] = 1  # White for goal
    
    plt.imshow(grid_colors, cmap='gray', alpha=0.3)
    plt.title('Learned Policy (Arrows show optimal actions)')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_learning_curve(rewards):
    """Plot the learning curve"""
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = 50
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    plt.plot(moving_avg)
    plt.title(f'Moving Average Reward (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training Q-Learning Agent in Grid World...")
    
    # Train the agent
    agent, rewards = train_agent(episodes=1000)
    
    # Visualize results
    env = GridWorld()
    env.render()
    
    # Show learning curve
    plot_learning_curve(rewards)
    
    # Show learned policy
    visualize_policy(agent, env)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 50:
        state = env.agent_pos
        valid_actions = env.get_valid_actions(state)
        action = agent.get_policy(state, valid_actions)
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: State {state} -> Action {action} -> Next State {next_state}, Reward: {reward}")
        
        if done:
            print(f"Goal reached in {steps} steps with total reward: {total_reward}")
            break
    
    if not done:
        print(f"Did not reach goal in {steps} steps. Total reward: {total_reward}")