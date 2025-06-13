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

class ActorNetwork(nn.Module):
    """Actor Network for policy"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class CriticNetwork(nn.Module):
    """Critic Network for value function"""
    
    def __init__(self, state_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCriticAgent:
    """Actor-Critic Agent"""
    
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Actor network (policy)
        self.actor = ActorNetwork(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic network (value function)
        self.critic = CriticNetwork(state_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def act(self, state):
        """Select action using actor network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)
    
    def evaluate_state(self, state):
        """Evaluate state using critic network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.critic(state_tensor)
    
    def update(self, state, action_log_prob, reward, next_state, done):
        """Update actor and critic networks"""
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        
        # Critic update
        current_value = self.critic(state_tensor)
        
        if done:
            target_value = reward_tensor
        else:
            next_value = self.critic(next_state_tensor).detach()
            target_value = reward_tensor + self.gamma * next_value
        
        # Compute TD error (advantage)
        advantage = target_value - current_value
        
        # Critic loss
        critic_loss = F.mse_loss(current_value, target_value.detach())
        
        # Actor loss
        actor_loss = -action_log_prob * advantage.detach()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), advantage.item()

def train_actor_critic(env_name='MountainCar-v0', episodes=2000, max_steps=200):
    """Train Actor-Critic agent"""
    
    # Create environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = ActorCriticAgent(state_size, action_size)
    
    # Training metrics
    scores = []
    actor_losses = []
    critic_losses = []
    advantages = []
    
    print(f"Training Actor-Critic on {env_name}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_actor_losses = []
        episode_critic_losses = []
        episode_advantages = []
        
        for step in range(max_steps):
            # Select action
            action, action_log_prob = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update agent
            actor_loss, critic_loss, advantage = agent.update(
                state, action_log_prob, reward, next_state, done
            )
            
            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)
            episode_advantages.append(advantage)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Store episode metrics
        scores.append(total_reward)
        actor_losses.append(np.mean(episode_actor_losses))
        critic_losses.append(np.mean(episode_critic_losses))
        advantages.append(np.mean(episode_advantages))
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode:4d} | Avg Score: {avg_score:6.2f} | "
                  f"Actor Loss: {actor_losses[-1]:8.4f} | "
                  f"Critic Loss: {critic_losses[-1]:8.4f}")
        
        # Check if solved (Mountain Car is solved when avg score > -110)
        if len(scores) >= 100 and np.mean(scores[-100:]) > -110:
            print(f"\nEnvironment solved in {episode} episodes!")
            print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")
            break
    
    env.close()
    return agent, scores, actor_losses, critic_losses, advantages

def test_agent(agent, env_name='MountainCar-v0', episodes=10, render=False):
    """Test trained agent"""
    env = gym.make(env_name)
    test_scores = []
    
    print(f"\nTesting agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 200:  # Max steps for Mountain Car
            if render:
                env.render()
            
            # Use actor network for action selection (greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
                action = action_probs.argmax().item()
            
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

def plot_training_results(scores, actor_losses, critic_losses, advantages):
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
    
    ax1.axhline(y=-110, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Actor and Critic losses
    ax2 = axes[0, 1]
    ax2.plot(actor_losses, label='Actor Loss', alpha=0.8)
    ax2.plot(critic_losses, label='Critic Loss', alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Advantages
    ax3 = axes[1, 0]
    ax3.plot(advantages, alpha=0.8)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Advantage')
    ax3.set_title('Advantage Values')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Score distribution
    ax4 = axes[1, 1]
    ax4.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scores):.1f}')
    ax4.axvline(x=-110, color='green', linestyle='--', label='Solved: -110')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_learned_functions(agent, env_name='MountainCar-v0'):
    """Analyze learned value function and policy"""
    
    # Create a grid of states
    position_range = np.linspace(-1.2, 0.6, 50)
    velocity_range = np.linspace(-0.07, 0.07, 50)
    
    positions, velocities = np.meshgrid(position_range, velocity_range)
    states = np.column_stack([positions.ravel(), velocities.ravel()])
    
    # Get value function and policy
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        values = agent.critic(state_tensor).numpy().flatten()
        action_probs = agent.actor(state_tensor).numpy()
        actions = action_probs.argmax(axis=1)
    
    # Reshape for plotting
    values = values.reshape(positions.shape)
    actions = actions.reshape(positions.shape)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Value function
    ax1 = axes[0]
    im1 = ax1.contourf(positions, velocities, values, levels=20, cmap='viridis')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Learned Value Function')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Policy
    ax2 = axes[1]
    im2 = ax2.contourf(positions, velocities, actions, levels=3, cmap='RdYlBu')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Learned Policy (0=Left, 1=Nothing, 2=Right)')
    plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    
    # Plot 3: Policy arrows
    ax3 = axes[2]
    # Subsample for arrow plot
    step = 5
    pos_sub = positions[::step, ::step]
    vel_sub = velocities[::step, ::step]
    actions_sub = actions[::step, ::step]
    
    # Convert actions to arrow directions
    u = np.zeros_like(pos_sub)
    v = np.zeros_like(vel_sub)
    
    # Action 0: Left (negative position change)
    mask_left = actions_sub == 0
    u[mask_left] = -0.02
    
    # Action 2: Right (positive position change)
    mask_right = actions_sub == 2
    u[mask_right] = 0.02
    
    # Action 1: Nothing (no change) - arrows will be zero
    
    ax3.quiver(pos_sub, vel_sub, u, v, actions_sub, cmap='RdYlBu', scale=1)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Policy Direction (Arrows)')
    ax3.set_xlim(-1.2, 0.6)
    ax3.set_ylim(-0.07, 0.07)
    
    plt.tight_layout()
    plt.show()

def compare_with_random_policy(agent, env_name='MountainCar-v0', episodes=100):
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
                action_probs = agent.actor(state_tensor)
                action = action_probs.argmax().item()
            
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
    plt.hist(trained_scores, bins=20, alpha=0.7, label='Trained Agent', color='blue')
    plt.hist(random_scores, bins=20, alpha=0.7, label='Random Policy', color='red')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([trained_scores, random_scores], labels=['Trained Agent', 'Random Policy'])
    plt.ylabel('Score')
    plt.title('Score Comparison (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Trained Agent - Mean: {np.mean(trained_scores):.2f}, Std: {np.std(trained_scores):.2f}")
    print(f"Random Policy - Mean: {np.mean(random_scores):.2f}, Std: {np.std(random_scores):.2f}")
    
    return trained_scores, random_scores

if __name__ == "__main__":
    print("Actor-Critic for Mountain Car")
    print("=" * 40)
    
    # Train the agent
    agent, scores, actor_losses, critic_losses, advantages = train_actor_critic(episodes=1500)
    
    # Plot training results
    plot_training_results(scores, actor_losses, critic_losses, advantages)
    
    # Test the trained agent
    test_scores = test_agent(agent, episodes=10)
    
    # Analyze learned functions
    print("\nAnalyzing learned value function and policy...")
    analyze_learned_functions(agent)
    
    # Compare with random policy
    print("\nComparing with random policy...")
    trained_scores, random_scores = compare_with_random_policy(agent, episodes=50)
    
    # Save the trained model
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, 'actor_critic_mountain_car.pth')
    
    print("\nModel saved as 'actor_critic_mountain_car.pth'")
    print(f"Training completed!")
    print(f"Final average score: {np.mean(scores[-100:]):.2f}")
    print(f"Test average score: {np.mean(test_scores):.2f}")
    
    # Check if solved
    if np.mean(scores[-100:]) > -110:
        print("Environment SOLVED! 🎉")
    else:
        print("Environment not fully solved, but significant improvement achieved.")