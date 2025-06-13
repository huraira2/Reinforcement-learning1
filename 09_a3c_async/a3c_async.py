import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
import matplotlib.pyplot as plt
import threading
import time
from collections import deque
import multiprocessing as mp

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ActorCriticNetwork(nn.Module):
    """Shared Actor-Critic Network for A3C"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy logits
        policy_logits = self.actor_head(x)
        
        # State value
        value = self.critic_head(x)
        
        return policy_logits, value
    
    def get_action_and_value(self, state):
        """Get action and value for given state"""
        policy_logits, value = self.forward(state)
        
        # Create categorical distribution
        dist = Categorical(logits=policy_logits)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()

class A3CWorker:
    """A3C Worker for asynchronous learning"""
    
    def __init__(self, worker_id, global_network, optimizer, env_name='CartPole-v1',
                 gamma=0.99, max_steps=200, update_frequency=20):
        
        self.worker_id = worker_id
        self.env_name = env_name
        self.gamma = gamma
        self.max_steps = max_steps
        self.update_frequency = update_frequency
        
        # Create local environment
        self.env = gym.make(env_name)
        
        # Global network and optimizer
        self.global_network = global_network
        self.optimizer = optimizer
        
        # Local network (copy of global)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.local_network = ActorCriticNetwork(state_size, action_size)
        
        # Sync local network with global
        self.sync_with_global()
        
        # Episode storage
        self.reset_episode()
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def sync_with_global(self):
        """Synchronize local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def reset_episode(self):
        """Reset episode storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages"""
        returns = []
        advantages = []
        
        # Add next value for bootstrapping
        values = self.values + [next_value]
        
        # Compute returns
        G = next_value
        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + self.gamma * G * (1 - self.dones[i])
            returns.insert(0, G)
        
        # Compute advantages
        for i in range(len(self.rewards)):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        
        return returns, advantages
    
    def update_global_network(self, next_value=0):
        """Update global network using local gradients"""
        if len(self.states) == 0:
            return 0, 0
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass through local network
        policy_logits, values = self.local_network(states)
        
        # Actor loss
        dist = Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        entropy_bonus = 0.01 * entropy
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - entropy_bonus
        
        # Compute gradients
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), 40)
        
        # Copy gradients to global network
        for local_param, global_param in zip(self.local_network.parameters(), 
                                           self.global_network.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad
        
        # Update global network
        self.optimizer.step()
        
        # Sync local network with updated global network
        self.sync_with_global()
        
        # Reset episode storage
        self.reset_episode()
        
        return actor_loss.item(), critic_loss.item()
    
    def run_episode(self):
        """Run a single episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.max_steps):
            # Get action and value from local network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.local_network.get_action_and_value(state_tensor)
            
            # Take action in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Store transition
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
            self.dones.append(done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update global network periodically or at episode end
            if len(self.states) >= self.update_frequency or done:
                next_value = 0
                if not done:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    _, next_value = self.local_network(next_state_tensor)
                    next_value = next_value.item()
                
                self.update_global_network(next_value)
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return episode_reward, episode_length
    
    def run(self, num_episodes):
        """Run worker for specified number of episodes"""
        print(f"Worker {self.worker_id} starting...")
        
        for episode in range(num_episodes):
            episode_reward, episode_length = self.run_episode()
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Worker {self.worker_id} | Episode {episode} | "
                      f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f}")
        
        self.env.close()
        print(f"Worker {self.worker_id} finished.")

class A3CAgent:
    """A3C Agent coordinating multiple workers"""
    
    def __init__(self, env_name='CartPole-v1', num_workers=4, lr=1e-3):
        
        self.env_name = env_name
        self.num_workers = num_workers
        
        # Create temporary environment to get dimensions
        temp_env = gym.make(env_name)
        state_size = temp_env.observation_space.shape[0]
        action_size = temp_env.action_space.n
        temp_env.close()
        
        # Global network
        self.global_network = ActorCriticNetwork(state_size, action_size)
        self.global_network.share_memory()  # Enable sharing between processes
        
        # Global optimizer
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=lr)
        
        # Workers
        self.workers = []
        
    def train(self, episodes_per_worker=500):
        """Train using multiple workers"""
        
        print(f"Training A3C with {self.num_workers} workers")
        print(f"Episodes per worker: {episodes_per_worker}")
        print("-" * 50)
        
        # Create workers
        for i in range(self.num_workers):
            worker = A3CWorker(i, self.global_network, self.optimizer, self.env_name)
            self.workers.append(worker)
        
        # Create threads for workers
        threads = []
        for worker in self.workers:
            thread = threading.Thread(target=worker.run, args=(episodes_per_worker,))
            threads.append(thread)
        
        # Start all workers
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all workers to finish
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
        
        return self.workers
    
    def test(self, episodes=10, render=False):
        """Test the trained global network"""
        env = gym.make(self.env_name)
        test_scores = []
        
        print(f"\nTesting global network for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 500:
                if render and episode < 3:
                    env.render()
                
                # Use global network for action selection
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, _ = self.global_network(state_tensor)
                    action = policy_logits.argmax().item()
                
                state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            test_scores.append(total_reward)
            print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
        
        env.close()
        
        avg_score = np.mean(test_scores)
        print(f"Average test score: {avg_score:.2f}")
        return test_scores

def plot_worker_performance(workers):
    """Plot performance of individual workers"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Individual worker rewards
    ax1 = axes[0, 0]
    for i, worker in enumerate(workers):
        # Moving average
        window_size = 50
        if len(worker.episode_rewards) >= window_size:
            moving_avg = [np.mean(worker.episode_rewards[max(0, j-window_size+1):j+1]) 
                         for j in range(len(worker.episode_rewards))]
            ax1.plot(moving_avg, label=f'Worker {i}', alpha=0.8)
    
    ax1.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Worker Performance (Moving Average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined performance
    ax2 = axes[0, 1]
    all_rewards = []
    for worker in workers:
        all_rewards.extend(worker.episode_rewards)
    
    # Sort by episode order (approximately)
    episodes_per_chunk = 100
    combined_rewards = []
    for i in range(0, len(all_rewards), episodes_per_chunk):
        chunk = all_rewards[i:i+episodes_per_chunk]
        combined_rewards.append(np.mean(chunk))
    
    ax2.plot(combined_rewards, alpha=0.8, linewidth=2)
    ax2.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax2.set_xlabel('Episode Chunk')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Combined Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final performance distribution
    ax3 = axes[1, 0]
    final_rewards = []
    for worker in workers:
        if len(worker.episode_rewards) >= 100:
            final_rewards.extend(worker.episode_rewards[-100:])
    
    ax3.hist(final_rewards, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(final_rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(final_rewards):.1f}')
    ax3.axvline(x=195, color='green', linestyle='--', label='Solved: 195')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Final Performance Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Worker statistics
    ax4 = axes[1, 1]
    worker_means = [np.mean(worker.episode_rewards[-100:]) if len(worker.episode_rewards) >= 100 
                   else np.mean(worker.episode_rewards) for worker in workers]
    worker_stds = [np.std(worker.episode_rewards[-100:]) if len(worker.episode_rewards) >= 100 
                  else np.std(worker.episode_rewards) for worker in workers]
    
    x = range(len(workers))
    bars = ax4.bar(x, worker_means, yerr=worker_stds, alpha=0.8, capsize=5)
    ax4.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax4.set_xlabel('Worker ID')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Worker Final Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, worker_means):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def compare_with_single_agent():
    """Compare A3C with single-threaded agent"""
    
    print("Comparing A3C with single-threaded learning...")
    
    # Train A3C
    print("\nTraining A3C (4 workers)...")
    a3c_agent = A3CAgent(num_workers=4)
    a3c_workers = a3c_agent.train(episodes_per_worker=300)
    
    # Test A3C
    a3c_test_scores = a3c_agent.test(episodes=20)
    
    # Train single agent (simulate by using 1 worker)
    print("\nTraining single agent...")
    single_agent = A3CAgent(num_workers=1)
    single_workers = single_agent.train(episodes_per_worker=1200)  # Same total episodes
    
    # Test single agent
    single_test_scores = single_agent.test(episodes=20)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Learning curves
    ax1 = axes[0]
    
    # A3C combined performance
    a3c_all_rewards = []
    for worker in a3c_workers:
        a3c_all_rewards.extend(worker.episode_rewards)
    
    # Single agent performance
    single_rewards = single_workers[0].episode_rewards
    
    # Moving averages
    window_size = 50
    a3c_ma = [np.mean(a3c_all_rewards[max(0, i-window_size+1):i+1]) 
              for i in range(len(a3c_all_rewards))]
    single_ma = [np.mean(single_rewards[max(0, i-window_size+1):i+1]) 
                 for i in range(len(single_rewards))]
    
    ax1.plot(a3c_ma, label='A3C (4 workers)', alpha=0.8)
    ax1.plot(single_ma, label='Single Agent', alpha=0.8)
    ax1.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test performance
    ax2 = axes[1]
    ax2.hist(a3c_test_scores, bins=15, alpha=0.7, label='A3C', density=True)
    ax2.hist(single_test_scores, bins=15, alpha=0.7, label='Single Agent', density=True)
    ax2.set_xlabel('Test Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Test Performance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance metrics
    ax3 = axes[2]
    
    metrics = ['Final Training\nScore', 'Test Score', 'Episodes to\nSolve']
    
    # Calculate episodes to solve (first time avg > 195)
    def episodes_to_solve(rewards, window=100):
        for i in range(window, len(rewards)):
            if np.mean(rewards[i-window:i]) >= 195:
                return i
        return len(rewards)
    
    a3c_final = np.mean(a3c_all_rewards[-100:])
    single_final = np.mean(single_rewards[-100:])
    
    a3c_values = [
        a3c_final,
        np.mean(a3c_test_scores),
        episodes_to_solve(a3c_all_rewards) / 4  # Divide by number of workers
    ]
    single_values = [
        single_final,
        np.mean(single_test_scores),
        episodes_to_solve(single_rewards)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, a3c_values, width, label='A3C', alpha=0.8)
    bars2 = ax3.bar(x + width/2, single_values, width, label='Single Agent', alpha=0.8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Value')
    ax3.set_title('Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return a3c_agent, single_agent, a3c_test_scores, single_test_scores

def analyze_worker_diversity(workers):
    """Analyze diversity among workers"""
    
    print("\nAnalyzing worker diversity...")
    
    # Collect final policies from each worker
    worker_policies = []
    
    # Create test environment
    env = gym.make('CartPole-v1')
    
    # Sample some states
    test_states = []
    for _ in range(100):
        state = env.reset()
        test_states.append(state)
        for _ in range(np.random.randint(1, 50)):
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            if not done:
                test_states.append(state)
            else:
                break
    
    env.close()
    test_states = np.array(test_states[:200])
    
    # Get action probabilities for each worker's final policy
    for worker in workers:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(test_states)
            policy_logits, _ = worker.local_network(state_tensor)
            action_probs = F.softmax(policy_logits, dim=1).numpy()
            worker_policies.append(action_probs)
    
    # Calculate policy diversity
    policy_distances = []
    for i in range(len(workers)):
        for j in range(i+1, len(workers)):
            # KL divergence between policies
            kl_div = np.mean(np.sum(worker_policies[i] * 
                                  np.log(worker_policies[i] / (worker_policies[j] + 1e-8)), 
                                  axis=1))
            policy_distances.append(kl_div)
    
    print(f"Average policy KL divergence: {np.mean(policy_distances):.4f}")
    print(f"Policy diversity std: {np.std(policy_distances):.4f}")
    
    # Plot policy comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Action preferences for each worker
    ax1 = axes[0, 0]
    for i, worker_policy in enumerate(worker_policies):
        action_preferences = np.mean(worker_policy, axis=0)
        ax1.bar(np.arange(2) + i*0.15, action_preferences, width=0.15, 
               label=f'Worker {i}', alpha=0.8)
    
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Average Probability')
    ax1.set_title('Action Preferences by Worker')
    ax1.set_xticks([0.3, 1.3])
    ax1.set_xticklabels(['Left', 'Right'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Policy entropy for each worker
    ax2 = axes[0, 1]
    worker_entropies = []
    for worker_policy in worker_policies:
        entropy = -np.sum(worker_policy * np.log(worker_policy + 1e-8), axis=1)
        worker_entropies.append(np.mean(entropy))
    
    bars = ax2.bar(range(len(workers)), worker_entropies, alpha=0.8)
    ax2.set_xlabel('Worker ID')
    ax2.set_ylabel('Average Policy Entropy')
    ax2.set_title('Policy Entropy by Worker')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, entropy in zip(bars, worker_entropies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy:.3f}', ha='center', va='bottom')
    
    # Plot 3: Policy distance matrix
    ax3 = axes[1, 0]
    distance_matrix = np.zeros((len(workers), len(workers)))
    idx = 0
    for i in range(len(workers)):
        for j in range(i+1, len(workers)):
            distance_matrix[i, j] = policy_distances[idx]
            distance_matrix[j, i] = policy_distances[idx]
            idx += 1
    
    im = ax3.imshow(distance_matrix, cmap='viridis')
    ax3.set_xlabel('Worker ID')
    ax3.set_ylabel('Worker ID')
    ax3.set_title('Policy Distance Matrix (KL Divergence)')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Performance vs diversity
    ax4 = axes[1, 1]
    worker_performance = [np.mean(worker.episode_rewards[-100:]) 
                         if len(worker.episode_rewards) >= 100 
                         else np.mean(worker.episode_rewards) 
                         for worker in workers]
    
    ax4.scatter(worker_entropies, worker_performance, s=100, alpha=0.8)
    for i, (entropy, perf) in enumerate(zip(worker_entropies, worker_performance)):
        ax4.annotate(f'W{i}', (entropy, perf), xytext=(5, 5), 
                    textcoords='offset points')
    
    ax4.set_xlabel('Policy Entropy')
    ax4.set_ylabel('Performance')
    ax4.set_title('Performance vs Policy Diversity')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("A3C (Asynchronous Advantage Actor-Critic)")
    print("=" * 50)
    
    # Train A3C agent
    agent = A3CAgent(num_workers=4)
    workers = agent.train(episodes_per_worker=400)
    
    # Plot worker performance
    plot_worker_performance(workers)
    
    # Test the trained agent
    test_scores = agent.test(episodes=20)
    
    # Analyze worker diversity
    analyze_worker_diversity(workers)
    
    # Compare with single agent
    print("\n" + "=" * 50)
    a3c_agent, single_agent, a3c_test_scores, single_test_scores = compare_with_single_agent()
    
    # Save the trained model
    torch.save(agent.global_network.state_dict(), 'a3c_cartpole.pth')
    
    print("\nModel saved as 'a3c_cartpole.pth'")
    print(f"Training completed!")
    
    # Final statistics
    all_rewards = []
    for worker in workers:
        all_rewards.extend(worker.episode_rewards)
    
    print(f"A3C Final average score: {np.mean(all_rewards[-400:]):.2f}")
    print(f"A3C Test average score: {np.mean(test_scores):.2f}")
    
    # Check if solved
    if np.mean(all_rewards[-400:]) >= 195:
        print("Environment SOLVED with A3C! 🎉")
    else:
        print("Environment not fully solved, but significant improvement achieved.")
    
    # Performance comparison
    print(f"\nComparison Results:")
    print(f"A3C Test Score: {np.mean(a3c_test_scores):.2f}")
    print(f"Single Agent Test Score: {np.mean(single_test_scores):.2f}")
    
    if np.mean(a3c_test_scores) > np.mean(single_test_scores):
        improvement = ((np.mean(a3c_test_scores) - np.mean(single_test_scores)) / 
                      np.mean(single_test_scores)) * 100
        print(f"🏆 A3C wins with {improvement:.1f}% improvement!")
    else:
        print("Single agent performed better in this run.")