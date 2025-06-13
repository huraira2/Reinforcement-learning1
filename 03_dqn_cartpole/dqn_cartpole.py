import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from collections import deque, namedtuple

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with Experience Replay and Target Network"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64, target_update=100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.step_count = 0
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

def train_dqn(episodes=1000, max_steps=500):
    """Train DQN agent on CartPole environment"""
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    scores = []
    losses = []
    epsilons = []
    
    print(f"Training DQN on CartPole-v1")
    print(f"State size: {state_size}, Action size: {action_size}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        epsilons.append(agent.epsilon)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode:4d} | Avg Score: {avg_score:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Loss: {losses[-1]:.4f}")
        
        # Check if solved
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195:
            print(f"\nEnvironment solved in {episode} episodes!")
            print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")
            break
    
    env.close()
    return agent, scores, losses, epsilons

def test_agent(agent, episodes=10, render=False):
    """Test trained agent"""
    env = gym.make('CartPole-v1')
    test_scores = []
    
    print(f"\nTesting agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            action = agent.act(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        test_scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
    
    env.close()
    
    avg_score = np.mean(test_scores)
    print(f"\nAverage test score: {avg_score:.2f}")
    return test_scores

def plot_training_results(scores, losses, epsilons):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Scores over episodes
    ax1 = axes[0, 0]
    ax1.plot(scores, alpha=0.6)
    
    # Moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = []
        for i in range(len(scores)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(scores[start_idx:i+1]))
        ax1.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
        ax1.legend()
    
    ax1.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Loss over episodes
    ax2 = axes[0, 1]
    ax2.plot(losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    ax3 = axes[1, 0]
    ax3.plot(epsilons)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Epsilon Decay')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Score distribution
    ax4 = axes[1, 1]
    ax4.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scores):.1f}')
    ax4.axvline(x=195, color='green', linestyle='--', label='Solved: 195')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_q_values(agent):
    """Analyze learned Q-values for different states"""
    env = gym.make('CartPole-v1')
    
    # Sample some states
    states = []
    for _ in range(1000):
        env.reset()
        for _ in range(random.randint(1, 100)):
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            if not done:
                states.append(state)
            else:
                break
    
    env.close()
    
    if not states:
        print("No valid states collected for analysis")
        return
    
    states = np.array(states[:100])  # Take first 100 states
    
    # Get Q-values for these states
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        q_values = agent.q_network(state_tensor).numpy()
    
    # Plot Q-values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q-values for action 0 (left)
    ax1 = axes[0, 0]
    ax1.scatter(states[:, 0], q_values[:, 0], alpha=0.6)
    ax1.set_xlabel('Cart Position')
    ax1.set_ylabel('Q-value (Left)')
    ax1.set_title('Q-values for Left Action vs Cart Position')
    ax1.grid(True, alpha=0.3)
    
    # Q-values for action 1 (right)
    ax2 = axes[0, 1]
    ax2.scatter(states[:, 0], q_values[:, 1], alpha=0.6, color='orange')
    ax2.set_xlabel('Cart Position')
    ax2.set_ylabel('Q-value (Right)')
    ax2.set_title('Q-values for Right Action vs Cart Position')
    ax2.grid(True, alpha=0.3)
    
    # Q-value difference (right - left)
    ax3 = axes[1, 0]
    q_diff = q_values[:, 1] - q_values[:, 0]
    ax3.scatter(states[:, 0], q_diff, alpha=0.6, color='green')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Cart Position')
    ax3.set_ylabel('Q-value Difference (Right - Left)')
    ax3.set_title('Action Preference vs Cart Position')
    ax3.grid(True, alpha=0.3)
    
    # Q-values vs pole angle
    ax4 = axes[1, 1]
    ax4.scatter(states[:, 2], q_values[:, 0], alpha=0.6, label='Left', s=20)
    ax4.scatter(states[:, 2], q_values[:, 1], alpha=0.6, label='Right', s=20)
    ax4.set_xlabel('Pole Angle')
    ax4.set_ylabel('Q-value')
    ax4.set_title('Q-values vs Pole Angle')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Deep Q-Network (DQN) for CartPole")
    print("=" * 40)
    
    # Train the agent
    agent, scores, losses, epsilons = train_dqn(episodes=1000)
    
    # Plot training results
    plot_training_results(scores, losses, epsilons)
    
    # Test the trained agent
    test_scores = test_agent(agent, episodes=10)
    
    # Analyze learned Q-values
    print("\nAnalyzing learned Q-values...")
    analyze_q_values(agent)
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'dqn_cartpole_model.pth')
    print("\nModel saved as 'dqn_cartpole_model.pth'")
    
    print(f"\nTraining completed!")
    print(f"Final average score: {np.mean(scores[-100:]):.2f}")
    print(f"Test average score: {np.mean(test_scores):.2f}")