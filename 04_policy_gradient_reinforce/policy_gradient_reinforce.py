import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PolicyNetwork(nn.Module):
    """Policy Network for REINFORCE"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ValueNetwork(nn.Module):
    """Value Network for baseline"""
    
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class REINFORCEAgent:
    """REINFORCE Agent with optional baseline"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, use_baseline=True):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Value network (baseline)
        if use_baseline:
            self.value_net = ValueNetwork(state_size)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Episode storage
        self.reset_episode()
        
    def reset_episode(self):
        """Reset episode storage"""
        self.log_probs = []
        self.rewards = []
        self.states = []
        
    def act(self, state):
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store log probability and state
        self.log_probs.append(dist.log_prob(action))
        self.states.append(state)
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def compute_returns(self):
        """Compute discounted returns"""
        returns = []
        G = 0
        
        # Compute returns backwards
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return torch.FloatTensor(returns)
    
    def update(self):
        """Update policy using REINFORCE"""
        if not self.rewards:
            return 0, 0
        
        # Compute returns
        returns = self.compute_returns()
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = 0
        value_loss = 0
        
        if self.use_baseline:
            # Compute baseline values
            states_tensor = torch.FloatTensor(self.states)
            baselines = self.value_net(states_tensor).squeeze()
            
            # Compute advantages
            advantages = returns - baselines
            
            # Policy loss with baseline
            for log_prob, advantage in zip(self.log_probs, advantages):
                policy_loss += -log_prob * advantage.detach()
            
            # Value loss
            value_loss = F.mse_loss(baselines, returns)
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optimizer.step()
            
        else:
            # Policy loss without baseline
            for log_prob, G in zip(self.log_probs, returns):
                policy_loss += -log_prob * G
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Reset episode storage
        self.reset_episode()
        
        return policy_loss.item(), value_loss.item() if self.use_baseline else 0

def train_reinforce(env_name='CartPole-v1', episodes=2000, max_steps=500, 
                   use_baseline=True, lr=1e-3):
    """Train REINFORCE agent"""
    
    # Create environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = REINFORCEAgent(state_size, action_size, lr=lr, use_baseline=use_baseline)
    
    # Training metrics
    scores = []
    policy_losses = []
    value_losses = []
    
    print(f"Training REINFORCE on {env_name}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Using baseline: {use_baseline}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        # Run episode
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update agent
        policy_loss, value_loss = agent.update()
        
        # Store metrics
        scores.append(total_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode:4d} | Avg Score: {avg_score:6.2f} | "
                  f"Policy Loss: {policy_loss:8.4f} | Value Loss: {value_loss:8.4f}")
        
        # Check if solved
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195:
            print(f"\nEnvironment solved in {episode} episodes!")
            print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")
            break
    
    env.close()
    return agent, scores, policy_losses, value_losses

def compare_with_without_baseline():
    """Compare REINFORCE with and without baseline"""
    
    print("Comparing REINFORCE with and without baseline...")
    
    # Train without baseline
    print("\nTraining without baseline:")
    agent_no_baseline, scores_no_baseline, _, _ = train_reinforce(
        episodes=1000, use_baseline=False, lr=1e-3
    )
    
    # Train with baseline
    print("\nTraining with baseline:")
    agent_baseline, scores_baseline, _, _ = train_reinforce(
        episodes=1000, use_baseline=True, lr=1e-3
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    
    # Moving averages
    window_size = 50
    
    def moving_average(data, window):
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    ma_no_baseline = moving_average(scores_no_baseline, window_size)
    ma_baseline = moving_average(scores_baseline, window_size)
    
    plt.plot(ma_no_baseline, label='Without Baseline', alpha=0.8)
    plt.plot(ma_baseline, label='With Baseline', alpha=0.8)
    plt.axhline(y=195, color='red', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title(f'REINFORCE Comparison (Moving Average, window={window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot variance
    plt.subplot(1, 2, 2)
    
    # Compute rolling variance
    def rolling_variance(data, window):
        variances = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i+1]
            variances.append(np.var(window_data))
        return variances
    
    var_no_baseline = rolling_variance(scores_no_baseline, window_size)
    var_baseline = rolling_variance(scores_baseline, window_size)
    
    plt.plot(var_no_baseline, label='Without Baseline', alpha=0.8)
    plt.plot(var_baseline, label='With Baseline', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Score Variance')
    plt.title(f'Score Variance Comparison (window={window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return agent_no_baseline, agent_baseline, scores_no_baseline, scores_baseline

def test_agent(agent, env_name='CartPole-v1', episodes=10, render=False):
    """Test trained agent"""
    env = gym.make(env_name)
    test_scores = []
    
    print(f"\nTesting agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            # Use policy network directly for testing
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = agent.policy_net(state_tensor)
                action = action_probs.argmax().item()  # Greedy action selection
            
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

def plot_training_results(scores, policy_losses, value_losses, use_baseline=True):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Scores over episodes
    ax1 = axes[0, 0]
    ax1.plot(scores, alpha=0.6)
    
    # Moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = [np.mean(scores[max(0, i-window_size+1):i+1]) 
                     for i in range(len(scores))]
        ax1.plot(moving_avg, color='red', linewidth=2, 
                label=f'Moving Average ({window_size})')
        ax1.legend()
    
    ax1.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Policy loss
    ax2 = axes[0, 1]
    ax2.plot(policy_losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Policy Loss')
    ax2.set_title('Policy Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Value loss (if using baseline)
    ax3 = axes[1, 0]
    if use_baseline and value_losses:
        ax3.plot(value_losses)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Value Loss')
        ax3.set_title('Value Loss (Baseline)')
    else:
        ax3.text(0.5, 0.5, 'No Baseline Used', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Value Loss (Baseline)')
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

def analyze_policy(agent, env_name='CartPole-v1'):
    """Analyze learned policy"""
    env = gym.make(env_name)
    
    # Sample states
    states = []
    for _ in range(1000):
        env.reset()
        for _ in range(np.random.randint(1, 100)):
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
    
    states = np.array(states[:200])  # Take first 200 states
    
    # Get action probabilities
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        action_probs = agent.policy_net(state_tensor).numpy()
    
    # Plot policy analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Action probabilities vs cart position
    ax1 = axes[0, 0]
    ax1.scatter(states[:, 0], action_probs[:, 0], alpha=0.6, label='Left', s=20)
    ax1.scatter(states[:, 0], action_probs[:, 1], alpha=0.6, label='Right', s=20)
    ax1.set_xlabel('Cart Position')
    ax1.set_ylabel('Action Probability')
    ax1.set_title('Action Probabilities vs Cart Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Action probabilities vs pole angle
    ax2 = axes[0, 1]
    ax2.scatter(states[:, 2], action_probs[:, 0], alpha=0.6, label='Left', s=20)
    ax2.scatter(states[:, 2], action_probs[:, 1], alpha=0.6, label='Right', s=20)
    ax2.set_xlabel('Pole Angle')
    ax2.set_ylabel('Action Probability')
    ax2.set_title('Action Probabilities vs Pole Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Policy entropy
    ax3 = axes[1, 0]
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
    ax3.scatter(states[:, 0], entropy, alpha=0.6, s=20)
    ax3.set_xlabel('Cart Position')
    ax3.set_ylabel('Policy Entropy')
    ax3.set_title('Policy Entropy vs Cart Position')
    ax3.grid(True, alpha=0.3)
    
    # Action preference
    ax4 = axes[1, 1]
    action_preference = action_probs[:, 1] - action_probs[:, 0]  # Right - Left
    ax4.scatter(states[:, 0], action_preference, alpha=0.6, s=20, c=states[:, 2], 
                cmap='viridis')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Cart Position')
    ax4.set_ylabel('Action Preference (Right - Left)')
    ax4.set_title('Action Preference (colored by pole angle)')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Pole Angle')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Policy Gradient REINFORCE for CartPole")
    print("=" * 40)
    
    # Compare with and without baseline
    agent_no_baseline, agent_baseline, scores_no_baseline, scores_baseline = compare_with_without_baseline()
    
    # Test both agents
    print("\nTesting agent without baseline:")
    test_scores_no_baseline = test_agent(agent_no_baseline, episodes=5)
    
    print("\nTesting agent with baseline:")
    test_scores_baseline = test_agent(agent_baseline, episodes=5)
    
    # Analyze the baseline agent's policy
    print("\nAnalyzing learned policy (with baseline)...")
    analyze_policy(agent_baseline)
    
    # Save models
    torch.save(agent_baseline.policy_net.state_dict(), 'reinforce_policy_baseline.pth')
    torch.save(agent_no_baseline.policy_net.state_dict(), 'reinforce_policy_no_baseline.pth')
    
    print("\nModels saved!")
    print(f"Baseline agent - Final avg score: {np.mean(scores_baseline[-100:]):.2f}")
    print(f"No baseline agent - Final avg score: {np.mean(scores_no_baseline[-100:]):.2f}")
    print(f"Baseline agent - Test avg score: {np.mean(test_scores_baseline):.2f}")
    print(f"No baseline agent - Test avg score: {np.mean(test_scores_no_baseline):.2f}")