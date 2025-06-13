import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats

class MultiArmedBandit:
    def __init__(self, n_arms=10, reward_means=None, reward_stds=None):
        self.n_arms = n_arms
        
        if reward_means is None:
            self.reward_means = np.random.normal(0, 1, n_arms)
        else:
            self.reward_means = np.array(reward_means)
        
        if reward_stds is None:
            self.reward_stds = np.ones(n_arms)
        else:
            self.reward_stds = np.array(reward_stds)
        
        self.optimal_arm = np.argmax(self.reward_means)
        self.optimal_reward = self.reward_means[self.optimal_arm]
    
    def pull_arm(self, arm):
        """Pull an arm and get a reward"""
        reward = np.random.normal(self.reward_means[arm], self.reward_stds[arm])
        return reward
    
    def get_optimal_reward(self):
        return self.optimal_reward

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_reward = 0
        self.total_actions = 0
    
    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.total_reward += reward
        self.total_actions += 1

class UCBAgent:
    def __init__(self, n_arms, c=2):
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_reward = 0
        self.total_actions = 0
    
    def select_action(self):
        if self.total_actions < self.n_arms:
            # Try each arm at least once
            return self.total_actions
        
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_actions) / (self.action_counts + 1e-8)
        )
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.total_reward += reward
        self.total_actions += 1

class ThompsonSamplingAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta distribution parameters for each arm
        self.alpha = np.ones(n_arms)  # Success count + 1
        self.beta = np.ones(n_arms)   # Failure count + 1
        self.total_reward = 0
        self.total_actions = 0
    
    def select_action(self):
        # Sample from beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, action, reward):
        # Convert reward to binary (success/failure)
        # Assuming rewards are roughly in [-2, 2] range
        success = 1 if reward > 0 else 0
        
        if success:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
        
        self.total_reward += reward
        self.total_actions += 1

class GreedyAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_reward = 0
        self.total_actions = 0
    
    def select_action(self):
        if self.total_actions < self.n_arms:
            # Try each arm at least once
            return self.total_actions
        return np.argmax(self.q_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
        self.total_reward += reward
        self.total_actions += 1

def run_experiment(bandit, agent, n_steps=1000):
    """Run a single experiment with given bandit and agent"""
    rewards = []
    regrets = []
    optimal_actions = []
    
    for step in range(n_steps):
        action = agent.select_action()
        reward = bandit.pull_arm(action)
        agent.update(action, reward)
        
        rewards.append(reward)
        regret = bandit.get_optimal_reward() - reward
        regrets.append(regret)
        optimal_actions.append(1 if action == bandit.optimal_arm else 0)
    
    return rewards, regrets, optimal_actions

def compare_algorithms(n_experiments=100, n_steps=1000, n_arms=10):
    """Compare different bandit algorithms"""
    
    # Create bandit problem
    reward_means = np.random.normal(0, 1, n_arms)
    bandit = MultiArmedBandit(n_arms, reward_means)
    
    print(f"Bandit arm means: {reward_means}")
    print(f"Optimal arm: {bandit.optimal_arm} with mean reward: {bandit.optimal_reward:.3f}")
    
    algorithms = {
        'Epsilon-Greedy (ε=0.1)': lambda: EpsilonGreedyAgent(n_arms, epsilon=0.1),
        'Epsilon-Greedy (ε=0.01)': lambda: EpsilonGreedyAgent(n_arms, epsilon=0.01),
        'UCB (c=2)': lambda: UCBAgent(n_arms, c=2),
        'Thompson Sampling': lambda: ThompsonSamplingAgent(n_arms),
        'Greedy': lambda: GreedyAgent(n_arms)
    }
    
    results = {}
    
    for alg_name, agent_factory in algorithms.items():
        print(f"\nRunning {alg_name}...")
        
        all_rewards = []
        all_regrets = []
        all_optimal_actions = []
        
        for exp in range(n_experiments):
            agent = agent_factory()
            rewards, regrets, optimal_actions = run_experiment(bandit, agent, n_steps)
            
            all_rewards.append(rewards)
            all_regrets.append(regrets)
            all_optimal_actions.append(optimal_actions)
        
        # Calculate statistics
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_regrets = np.mean(all_regrets, axis=0)
        avg_optimal_actions = np.mean(all_optimal_actions, axis=0)
        
        results[alg_name] = {
            'rewards': avg_rewards,
            'regrets': avg_regrets,
            'optimal_actions': avg_optimal_actions,
            'cumulative_regret': np.cumsum(avg_regrets)
        }
        
        print(f"Final average reward: {avg_rewards[-1]:.3f}")
        print(f"Final cumulative regret: {np.sum(avg_regrets):.3f}")
        print(f"Final optimal action %: {avg_optimal_actions[-1]*100:.1f}%")
    
    return results, bandit

def plot_results(results, bandit):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average Reward over Time
    ax1 = axes[0, 0]
    for alg_name, data in results.items():
        ax1.plot(data['rewards'], label=alg_name, alpha=0.8)
    ax1.axhline(y=bandit.optimal_reward, color='red', linestyle='--', 
                label=f'Optimal ({bandit.optimal_reward:.3f})')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Regret
    ax2 = axes[0, 1]
    for alg_name, data in results.items():
        ax2.plot(data['cumulative_regret'], label=alg_name, alpha=0.8)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Cumulative Regret over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimal Action Percentage
    ax3 = axes[1, 0]
    for alg_name, data in results.items():
        # Smooth the optimal action percentage
        window_size = 50
        smoothed = np.convolve(data['optimal_actions'], 
                              np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(window_size-1, len(data['optimal_actions'])), 
                smoothed, label=alg_name, alpha=0.8)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Optimal Action %')
    ax3.set_title('Percentage of Optimal Actions (Smoothed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Performance Comparison
    ax4 = axes[1, 1]
    final_rewards = [data['rewards'][-1] for data in results.values()]
    final_regrets = [data['cumulative_regret'][-1] for data in results.values()]
    
    x_pos = np.arange(len(results))
    bars = ax4.bar(x_pos, final_rewards, alpha=0.7)
    ax4.axhline(y=bandit.optimal_reward, color='red', linestyle='--', 
                label=f'Optimal ({bandit.optimal_reward:.3f})')
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Final Average Reward')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(list(results.keys()), rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars, final_rewards):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{reward:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def demonstrate_single_algorithm():
    """Demonstrate a single algorithm in detail"""
    print("Demonstrating Epsilon-Greedy Algorithm...")
    
    # Create a simple 3-arm bandit
    bandit = MultiArmedBandit(3, reward_means=[0.1, 0.5, 0.3])
    agent = EpsilonGreedyAgent(3, epsilon=0.1)
    
    print(f"True arm means: {bandit.reward_means}")
    print(f"Optimal arm: {bandit.optimal_arm}")
    
    # Run for 100 steps with detailed output
    for step in range(100):
        action = agent.select_action()
        reward = bandit.pull_arm(action)
        agent.update(action, reward)
        
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Estimated Q-values: {agent.q_values}")
            print(f"  Action counts: {agent.action_counts}")
            print(f"  Selected action: {action}, Reward: {reward:.3f}")

if __name__ == "__main__":
    print("Multi-Armed Bandit Comparison")
    print("=" * 40)
    
    # Run detailed demonstration
    demonstrate_single_algorithm()
    
    print("\n" + "=" * 40)
    print("Running full comparison...")
    
    # Run full comparison
    results, bandit = compare_algorithms(n_experiments=50, n_steps=1000, n_arms=10)
    
    # Plot results
    plot_results(results, bandit)
    
    # Print summary
    print("\nSummary:")
    print("-" * 30)
    for alg_name, data in results.items():
        final_reward = data['rewards'][-1]
        total_regret = data['cumulative_regret'][-1]
        optimal_pct = data['optimal_actions'][-1] * 100
        
        print(f"{alg_name}:")
        print(f"  Final avg reward: {final_reward:.3f}")
        print(f"  Total regret: {total_regret:.1f}")
        print(f"  Optimal action %: {optimal_pct:.1f}%")
        print()