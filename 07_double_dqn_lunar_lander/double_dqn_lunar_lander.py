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
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

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

class DoubleDQNAgent:
    """Double DQN Agent"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=100000, batch_size=64, target_update=100,
                 double_dqn=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        
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
            return None
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        if self.double_dqn:
            # Double DQN: Use main network for action selection, target network for evaluation
            next_actions = self.q_network(next_states).argmax(1).detach()
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        else:
            # Standard DQN: Use target network for both action selection and evaluation
            next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

def train_agent(agent_type='double_dqn', episodes=2000, max_steps=1000):
    """Train DQN or Double DQN agent on Lunar Lander"""
    
    # Create environment
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    double_dqn = (agent_type == 'double_dqn')
    agent = DoubleDQNAgent(state_size, action_size, double_dqn=double_dqn)
    
    # Training metrics
    scores = []
    losses = []
    epsilons = []
    
    print(f"Training {'Double DQN' if double_dqn else 'Standard DQN'} on LunarLander-v2")
    print(f"State size: {state_size}, Action size: {action_size}")
    print("-" * 60)
    
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
            print(f"Episode {episode:4d} | Avg Score: {avg_score:7.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Loss: {losses[-1]:.4f}")
        
        # Check if solved (Lunar Lander is solved when avg score > 200)
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 200:
            print(f"\nEnvironment solved in {episode} episodes!")
            print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")
            break
    
    env.close()
    return agent, scores, losses, epsilons

def compare_dqn_variants():
    """Compare Standard DQN vs Double DQN"""
    
    print("Comparing Standard DQN vs Double DQN")
    print("=" * 60)
    
    # Train Standard DQN
    print("\nTraining Standard DQN...")
    dqn_agent, dqn_scores, dqn_losses, dqn_epsilons = train_agent('dqn', episodes=1500)
    
    # Train Double DQN
    print("\nTraining Double DQN...")
    ddqn_agent, ddqn_scores, ddqn_losses, ddqn_epsilons = train_agent('double_dqn', episodes=1500)
    
    # Test both agents
    print("\nTesting Standard DQN:")
    dqn_test_scores = test_agent(dqn_agent, episodes=100)
    
    print("\nTesting Double DQN:")
    ddqn_test_scores = test_agent(ddqn_agent, episodes=100)
    
    # Plot comparison
    plot_comparison(dqn_scores, ddqn_scores, dqn_losses, ddqn_losses,
                   dqn_epsilons, ddqn_epsilons, dqn_test_scores, ddqn_test_scores)
    
    return (dqn_agent, ddqn_agent, dqn_scores, ddqn_scores, 
            dqn_test_scores, ddqn_test_scores)

def test_agent(agent, episodes=10, render=False):
    """Test trained agent"""
    env = gym.make('LunarLander-v2')
    test_scores = []
    
    print(f"Testing agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:
            if render and episode < 3:
                env.render()
            
            action = agent.act(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        test_scores.append(total_reward)
        if episode < 10:
            print(f"Test Episode {episode + 1}: Score = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    avg_score = np.mean(test_scores)
    print(f"Average test score: {avg_score:.2f}")
    return test_scores

def plot_comparison(dqn_scores, ddqn_scores, dqn_losses, ddqn_losses,
                   dqn_epsilons, ddqn_epsilons, dqn_test_scores, ddqn_test_scores):
    """Plot comparison between DQN variants"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Moving average function
    def moving_average(data, window_size=100):
        return [np.mean(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]
    
    # Plot 1: Training scores comparison
    ax1 = axes[0, 0]
    dqn_ma = moving_average(dqn_scores)
    ddqn_ma = moving_average(ddqn_scores)
    
    ax1.plot(dqn_ma, label='Standard DQN', alpha=0.8)
    ax1.plot(ddqn_ma, label='Double DQN', alpha=0.8)
    ax1.axhline(y=200,  color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Training Scores (Moving Average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training losses comparison
    ax2 = axes[0, 1]
    dqn_loss_ma = moving_average(dqn_losses, window_size=50)
    ddqn_loss_ma = moving_average(ddqn_losses, window_size=50)
    
    ax2.plot(dqn_loss_ma, label='Standard DQN', alpha=0.8)
    ax2.plot(ddqn_loss_ma, label='Double DQN', alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Loss')
    ax2.set_title('Training Loss (Moving Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay comparison
    ax3 = axes[0, 2]
    ax3.plot(dqn_epsilons, label='Standard DQN', alpha=0.8)
    ax3.plot(ddqn_epsilons, label='Double DQN', alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Epsilon Decay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Test scores comparison
    ax4 = axes[1, 0]
    ax4.hist(dqn_test_scores, bins=20, alpha=0.7, label='Standard DQN', density=True)
    ax4.hist(ddqn_test_scores, bins=20, alpha=0.7, label='Double DQN', density=True)
    ax4.axvline(x=200, color='green', linestyle='--', label='Solved Threshold')
    ax4.set_xlabel('Test Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Test Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance metrics
    ax5 = axes[1, 1]
    
    metrics = ['Final Training\nScore', 'Test Score', 'Success Rate\n(>200)']
    dqn_values = [
        np.mean(dqn_scores[-100:]),
        np.mean(dqn_test_scores),
        np.mean([s > 200 for s in dqn_test_scores]) * 100
    ]
    ddqn_values = [
        np.mean(ddqn_scores[-100:]),
        np.mean(ddqn_test_scores),
        np.mean([s > 200 for s in ddqn_test_scores]) * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, dqn_values, width, label='Standard DQN', alpha=0.8)
    bars2 = ax5.bar(x + width/2, ddqn_values, width, label='Double DQN', alpha=0.8)
    
    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Value')
    ax5.set_title('Performance Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    # Plot 6: Learning curves comparison
    ax6 = axes[1, 2]
    
    # Calculate cumulative best scores
    dqn_cumulative_best = np.maximum.accumulate(dqn_ma)
    ddqn_cumulative_best = np.maximum.accumulate(ddqn_ma)
    
    ax6.plot(dqn_cumulative_best, label='Standard DQN', alpha=0.8)
    ax6.plot(ddqn_cumulative_best, label='Double DQN', alpha=0.8)
    ax6.axhline(y=200, color='green', linestyle='--', label='Solved Threshold')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Best Score So Far')
    ax6.set_title('Learning Progress (Cumulative Best)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_q_values(agent, agent_name="Agent"):
    """Analyze Q-values for different states"""
    env = gym.make('LunarLander-v2')
    
    # Collect some states
    states = []
    for _ in range(100):
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
    
    states = np.array(states[:200])  # Take first 200 states
    
    # Get Q-values for these states
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        q_values = agent.q_network(state_tensor).numpy()
    
    # Plot Q-value analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']
    
    # Plot 1: Q-values vs horizontal position
    ax1 = axes[0, 0]
    for action in range(4):
        ax1.scatter(states[:, 0], q_values[:, action], alpha=0.6, 
                   label=action_names[action], s=20)
    ax1.set_xlabel('Horizontal Position')
    ax1.set_ylabel('Q-value')
    ax1.set_title('Q-values vs Horizontal Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Q-values vs vertical position
    ax2 = axes[0, 1]
    for action in range(4):
        ax2.scatter(states[:, 1], q_values[:, action], alpha=0.6, 
                   label=action_names[action], s=20)
    ax2.set_xlabel('Vertical Position')
    ax2.set_ylabel('Q-value')
    ax2.set_title('Q-values vs Vertical Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action preferences
    ax3 = axes[1, 0]
    best_actions = q_values.argmax(axis=1)
    action_counts = np.bincount(best_actions, minlength=4)
    
    bars = ax3.bar(range(4), action_counts, alpha=0.8)
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Action Preferences')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(action_names, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Plot 4: Q-value distribution
    ax4 = axes[1, 1]
    ax4.hist(q_values.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(q_values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(q_values):.2f}')
    ax4.set_xlabel('Q-value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Q-value Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{agent_name} Q-value Analysis')
    plt.tight_layout()
    plt.show()

def demonstrate_landing(agent, agent_name="Agent"):
    """Demonstrate a landing attempt"""
    env = gym.make('LunarLander-v2')
    
    print(f"\nDemonstrating {agent_name} landing attempt:")
    print("-" * 50)
    
    state = env.reset()
    total_reward = 0
    step = 0
    action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']
    
    print(f"Initial state: x={state[0]:.3f}, y={state[1]:.3f}, "
          f"vx={state[2]:.3f}, vy={state[3]:.3f}")
    
    while step < 1000:
        action = agent.act(state, training=False)
        
        if step % 50 == 0:  # Print every 50 steps
            print(f"Step {step:3d}: Action={action_names[action]}, "
                  f"x={state[0]:.3f}, y={state[1]:.3f}, "
                  f"vx={state[2]:.3f}, vy={state[3]:.3f}")
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
        
        if done:
            print(f"Landing completed in {step} steps!")
            print(f"Final reward: {total_reward:.2f}")
            
            if total_reward >= 200:
                print("🎉 Successful landing!")
            elif total_reward >= 100:
                print("✅ Good landing!")
            elif total_reward >= 0:
                print("⚠️  Rough landing")
            else:
                print("💥 Crashed!")
            break
    
    env.close()
    return total_reward, step

if __name__ == "__main__":
    print("Double DQN vs Standard DQN on Lunar Lander")
    print("=" * 60)
    
    # Compare DQN variants
    (dqn_agent, ddqn_agent, dqn_scores, ddqn_scores, 
     dqn_test_scores, ddqn_test_scores) = compare_dqn_variants()
    
    # Analyze Q-values for both agents
    print("\nAnalyzing Standard DQN Q-values...")
    analyze_q_values(dqn_agent, "Standard DQN")
    
    print("\nAnalyzing Double DQN Q-values...")
    analyze_q_values(ddqn_agent, "Double DQN")
    
    # Demonstrate landings
    demonstrate_landing(dqn_agent, "Standard DQN")
    demonstrate_landing(ddqn_agent, "Double DQN")
    
    # Save models
    torch.save(dqn_agent.q_network.state_dict(), 'dqn_lunar_lander.pth')
    torch.save(ddqn_agent.q_network.state_dict(), 'double_dqn_lunar_lander.pth')
    
    print("\nModels saved!")
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    print(f"Standard DQN:")
    print(f"  Training - Final avg score: {np.mean(dqn_scores[-100:]):.2f}")
    print(f"  Testing  - Avg score: {np.mean(dqn_test_scores):.2f}")
    print(f"  Success rate (>200): {np.mean([s > 200 for s in dqn_test_scores])*100:.1f}%")
    
    print(f"\nDouble DQN:")
    print(f"  Training - Final avg score: {np.mean(ddqn_scores[-100:]):.2f}")
    print(f"  Testing  - Avg score: {np.mean(ddqn_test_scores):.2f}")
    print(f"  Success rate (>200): {np.mean([s > 200 for s in ddqn_test_scores])*100:.1f}%")
    
    # Determine winner
    dqn_score = np.mean(dqn_test_scores)
    ddqn_score = np.mean(ddqn_test_scores)
    
    if ddqn_score > dqn_score:
        improvement = ((ddqn_score - dqn_score) / abs(dqn_score)) * 100
        print(f"\n🏆 Double DQN wins with {ddqn_score:.2f} vs {dqn_score:.2f}")
        print(f"   Improvement: {improvement:.1f}%")
    elif dqn_score > ddqn_score:
        print(f"\n🏆 Standard DQN wins with {dqn_score:.2f} vs {ddqn_score:.2f}")
    else:
        print(f"\n🤝 It's a tie! Both scored {dqn_score:.2f}")