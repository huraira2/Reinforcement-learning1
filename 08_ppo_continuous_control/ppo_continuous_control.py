import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ActorNetwork(nn.Module):
    """Actor Network for continuous actions"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Mean and log standard deviation for Gaussian policy
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state):
        """Get action and its log probability"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        """Get log probability of given action"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return log_prob

class CriticNetwork(nn.Module):
    """Critic Network for value function"""
    
    def __init__(self, state_size, hidden_size=64):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        
        return value

class PPOAgent:
    """PPO Agent for continuous control"""
    
    def __init__(self, state_size, action_size, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 ppo_epochs=10, batch_size=64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Storage for trajectory
        self.reset_trajectory()
        
    def reset_trajectory(self):
        """Reset trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def act(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
            value = self.critic(state_tensor)
        
        return action.squeeze().numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Add next value for last state
        values = self.values + [next_value]
        
        # Compute advantages backwards
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_value=0):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return 0, 0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.FloatTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor loss (PPO clipped objective)
                new_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
        
        # Reset trajectory
        self.reset_trajectory()
        
        return total_actor_loss / self.ppo_epochs, total_critic_loss / self.ppo_epochs

def train_ppo(env_name='Pendulum-v1', episodes=1000, max_steps=200, update_frequency=2048):
    """Train PPO agent"""
    
    # Create environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    # Create agent
    agent = PPOAgent(state_size, action_size)
    
    # Training metrics
    scores = []
    actor_losses = []
    critic_losses = []
    
    print(f"Training PPO on {env_name}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Action space: {env.action_space}")
    print("-" * 60)
    
    episode = 0
    step_count = 0
    
    while episode < episodes:
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Get action
            action, log_prob, value = agent.act(state)
            
            # Clip action to environment bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            
            # Update policy
            if step_count % update_frequency == 0:
                next_value = 0
                if not done:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    with torch.no_grad():
                        next_value = agent.critic(next_state_tensor).item()
                
                actor_loss, critic_loss = agent.update(next_value)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            if done:
                break
        
        scores.append(episode_reward)
        episode += 1
        
        # Print progress
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode:4d} | Avg Score: {avg_score:7.2f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Actor Loss: {actor_losses[-1] if actor_losses else 0:.4f}")
        
        # Check if solved (Pendulum is considered solved at -200 or better)
        if len(scores) >= 100 and np.mean(scores[-100:]) >= -200:
            print(f"\nEnvironment solved in {episode} episodes!")
            print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")
            break
    
    env.close()
    return agent, scores, actor_losses, critic_losses

def test_agent(agent, env_name='Pendulum-v1', episodes=10, render=False):
    """Test trained agent"""
    env = gym.make(env_name)
    test_scores = []
    
    print(f"\nTesting agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 200:
            if render and episode < 3:
                env.render()
            
            # Get action (deterministic for testing)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(state_tensor)
                action = mean.squeeze().numpy()
            
            # Clip action
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        test_scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    avg_score = np.mean(test_scores)
    print(f"Average test score: {avg_score:.2f}")
    return test_scores

def plot_training_results(scores, actor_losses, critic_losses):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Scores over episodes
    ax1 = axes[0, 0]
    ax1.plot(scores, alpha=0.6)
    
    # Moving average
    window_size = 50
    if len(scores) >= window_size:
        moving_avg = [np.mean(scores[max(0, i-window_size+1):i+1]) 
                     for i in range(len(scores))]
        ax1.plot(moving_avg, color='red', linewidth=2, 
                label=f'Moving Average ({window_size})')
        ax1.legend()
    
    ax1.axhline(y=-200, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Actor loss
    ax2 = axes[0, 1]
    if actor_losses:
        ax2.plot(actor_losses, alpha=0.8)
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Actor Loss')
        ax2.set_title('Actor Loss')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Critic loss
    ax3 = axes[1, 0]
    if critic_losses:
        ax3.plot(critic_losses, alpha=0.8)
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Critic Loss')
        ax3.set_title('Critic Loss')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Score distribution
    ax4 = axes[1, 1]
    ax4.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scores):.1f}')
    ax4.axvline(x=-200, color='green', linestyle='--', label='Solved: -200')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_policy(agent, env_name='Pendulum-v1'):
    """Analyze learned policy"""
    env = gym.make(env_name)
    
    # Sample states
    states = []
    for _ in range(1000):
        state = env.reset()
        states.append(state)
        
        for _ in range(np.random.randint(1, 50)):
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            if not done:
                states.append(state)
            else:
                break
    
    env.close()
    
    if not states:
        print("No states collected for analysis")
        return
    
    states = np.array(states[:500])  # Take first 500 states
    
    # Get policy outputs
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        means, log_stds = agent.actor(state_tensor)
        values = agent.critic(state_tensor)
        
        means = means.numpy()
        stds = log_stds.exp().numpy()
        values = values.numpy().flatten()
    
    # Plot policy analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Action means vs angle
    ax1 = axes[0, 0]
    ax1.scatter(states[:, 0], means[:, 0], alpha=0.6, s=20)
    ax1.set_xlabel('Cos(θ)')
    ax1.set_ylabel('Action Mean')
    ax1.set_title('Policy Mean vs Cos(θ)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action stds vs angle
    ax2 = axes[0, 1]
    ax2.scatter(states[:, 0], stds[:, 0], alpha=0.6, s=20, c='orange')
    ax2.set_xlabel('Cos(θ)')
    ax2.set_ylabel('Action Std')
    ax2.set_title('Policy Std vs Cos(θ)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Value function vs angle
    ax3 = axes[1, 0]
    ax3.scatter(states[:, 0], values, alpha=0.6, s=20, c='green')
    ax3.set_xlabel('Cos(θ)')
    ax3.set_ylabel('Value')
    ax3.set_title('Value Function vs Cos(θ)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Action distribution
    ax4 = axes[1, 1]
    ax4.hist(means[:, 0], bins=30, alpha=0.7, label='Action Means', edgecolor='black')
    ax4.axvline(x=np.mean(means[:, 0]), color='red', linestyle='--', 
                label=f'Mean: {np.mean(means[:, 0]):.2f}')
    ax4.set_xlabel('Action')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Action Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_with_random_policy(agent, env_name='Pendulum-v1', episodes=50):
    """Compare trained agent with random policy"""
    
    env = gym.make(env_name)
    
    # Test trained agent
    trained_scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 200:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(state_tensor)
                action = mean.squeeze().numpy()
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        trained_scores.append(total_reward)
    
    # Test random policy
    random_scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 200:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        random_scores.append(total_reward)
    
    env.close()
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(trained_scores, bins=20, alpha=0.7, label='PPO Agent', color='blue')
    plt.hist(random_scores, bins=20, alpha=0.7, label='Random Policy', color='red')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([trained_scores, random_scores], labels=['PPO Agent', 'Random Policy'])
    plt.ylabel('Score')
    plt.title('Score Comparison (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"PPO Agent    - Mean: {np.mean(trained_scores):.2f}, Std: {np.std(trained_scores):.2f}")
    print(f"Random Policy - Mean: {np.mean(random_scores):.2f}, Std: {np.std(random_scores):.2f}")
    
    return trained_scores, random_scores

if __name__ == "__main__":
    print("PPO (Proximal Policy Optimization) for Continuous Control")
    print("=" * 60)
    
    # Train the agent
    agent, scores, actor_losses, critic_losses = train_ppo(episodes=800)
    
    # Plot training results
    plot_training_results(scores, actor_losses, critic_losses)
    
    # Test the trained agent
    test_scores = test_agent(agent, episodes=10)
    
    # Analyze learned policy
    print("\nAnalyzing learned policy...")
    analyze_policy(agent)
    
    # Compare with random policy
    print("\nComparing with random policy...")
    trained_scores, random_scores = compare_with_random_policy(agent, episodes=50)
    
    # Save the trained model
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, 'ppo_pendulum.pth')
    
    print("\nModel saved as 'ppo_pendulum.pth'")
    print(f"Training completed!")
    print(f"Final average score: {np.mean(scores[-100:]):.2f}")
    print(f"Test average score: {np.mean(test_scores):.2f}")
    
    # Check if solved
    if np.mean(scores[-100:]) >= -200:
        print("Environment SOLVED! 🎉")
    else:
        print("Environment not fully solved, but significant improvement achieved.")
    
    # Performance improvement
    improvement = ((np.mean(trained_scores) - np.mean(random_scores)) / 
                  abs(np.mean(random_scores))) * 100
    print(f"Performance improvement over random policy: {improvement:.1f}%")