import numpy as np
import gym
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class SARSAAgent:
    """SARSA Agent for on-policy learning"""
    
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table using defaultdict
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """SARSA update rule"""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        
        # SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_policy(self, state):
        """Get the greedy action for a state"""
        return np.argmax(self.q_table[state])

class QLearningAgent:
    """Q-Learning Agent for comparison (off-policy)"""
    
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table using defaultdict
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """Q-Learning update rule"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_policy(self, state):
        """Get the greedy action for a state"""
        return np.argmax(self.q_table[state])

def train_sarsa(episodes=10000, max_steps=200):
    """Train SARSA agent on Taxi environment"""
    
    # Create environment
    env = gym.make('Taxi-v3')
    n_actions = env.action_space.n
    
    # Create agent
    agent = SARSAAgent(n_actions)
    
    # Training metrics
    rewards_per_episode = []
    steps_per_episode = []
    epsilons = []
    
    print(f"Training SARSA on Taxi-v3")
    print(f"Action space size: {n_actions}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.get_action(state)  # Choose initial action
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Choose next action (important for SARSA)
            next_action = agent.get_action(next_state)
            
            # Update Q-value using SARSA
            agent.update(state, action, reward, next_state, next_action)
            
            # Move to next state and action
            state = next_state
            action = next_action
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilons.append(agent.epsilon)
        
        # Print progress
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            avg_steps = np.mean(steps_per_episode[-1000:])
            print(f"Episode {episode:5d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Steps: {avg_steps:6.2f} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    return agent, rewards_per_episode, steps_per_episode, epsilons

def train_qlearning(episodes=10000, max_steps=200):
    """Train Q-Learning agent for comparison"""
    
    # Create environment
    env = gym.make('Taxi-v3')
    n_actions = env.action_space.n
    
    # Create agent
    agent = QLearningAgent(n_actions)
    
    # Training metrics
    rewards_per_episode = []
    steps_per_episode = []
    epsilons = []
    
    print(f"Training Q-Learning on Taxi-v3")
    print(f"Action space size: {n_actions}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value using Q-Learning
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilons.append(agent.epsilon)
        
        # Print progress
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            avg_steps = np.mean(steps_per_episode[-1000:])
            print(f"Episode {episode:5d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Steps: {avg_steps:6.2f} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    return agent, rewards_per_episode, steps_per_episode, epsilons

def test_agent(agent, episodes=100, render=False):
    """Test trained agent"""
    env = gym.make('Taxi-v3')
    test_rewards = []
    test_steps = []
    
    print(f"\nTesting agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 200:  # Max steps
            if render and episode < 5:  # Render first 5 episodes
                env.render()
                print(f"State: {state}, Action: {agent.get_policy(state)}")
            
            action = agent.get_policy(state)  # Greedy action
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        
        if episode < 10:
            print(f"Test Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()
    
    avg_reward = np.mean(test_rewards)
    avg_steps = np.mean(test_steps)
    print(f"\nTest Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Success rate: {np.mean([r > 0 for r in test_rewards])*100:.1f}%")
    
    return test_rewards, test_steps

def compare_sarsa_qlearning():
    """Compare SARSA and Q-Learning"""
    
    print("Comparing SARSA vs Q-Learning")
    print("=" * 50)
    
    # Train both agents
    sarsa_agent, sarsa_rewards, sarsa_steps, sarsa_epsilons = train_sarsa(episodes=8000)
    qlearning_agent, qlearning_rewards, qlearning_steps, qlearning_epsilons = train_qlearning(episodes=8000)
    
    # Test both agents
    print("\nTesting SARSA agent:")
    sarsa_test_rewards, sarsa_test_steps = test_agent(sarsa_agent, episodes=100)
    
    print("\nTesting Q-Learning agent:")
    qlearning_test_rewards, qlearning_test_steps = test_agent(qlearning_agent, episodes=100)
    
    # Plot comparison
    plot_comparison(sarsa_rewards, qlearning_rewards, sarsa_steps, qlearning_steps,
                   sarsa_epsilons, qlearning_epsilons)
    
    return (sarsa_agent, qlearning_agent, 
            sarsa_rewards, qlearning_rewards,
            sarsa_test_rewards, qlearning_test_rewards)

def plot_comparison(sarsa_rewards, qlearning_rewards, sarsa_steps, qlearning_steps,
                   sarsa_epsilons, qlearning_epsilons):
    """Plot comparison between SARSA and Q-Learning"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Moving average function
    def moving_average(data, window_size=500):
        return [np.mean(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]
    
    # Plot 1: Rewards comparison
    ax1 = axes[0, 0]
    sarsa_ma = moving_average(sarsa_rewards)
    qlearning_ma = moving_average(qlearning_rewards)
    
    ax1.plot(sarsa_ma, label='SARSA', alpha=0.8)
    ax1.plot(qlearning_ma, label='Q-Learning', alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Rewards (Moving Average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps comparison
    ax2 = axes[0, 1]
    sarsa_steps_ma = moving_average(sarsa_steps)
    qlearning_steps_ma = moving_average(qlearning_steps)
    
    ax2.plot(sarsa_steps_ma, label='SARSA', alpha=0.8)
    ax2.plot(qlearning_steps_ma, label='Q-Learning', alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.set_title('Training Steps (Moving Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    ax3 = axes[0, 2]
    ax3.plot(sarsa_epsilons, label='SARSA', alpha=0.8)
    ax3.plot(qlearning_epsilons, label='Q-Learning', alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Epsilon Decay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final reward distribution
    ax4 = axes[1, 0]
    ax4.hist(sarsa_rewards[-1000:], bins=30, alpha=0.7, label='SARSA', density=True)
    ax4.hist(qlearning_rewards[-1000:], bins=30, alpha=0.7, label='Q-Learning', density=True)
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Density')
    ax4.set_title('Final Reward Distribution (Last 1000 episodes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Final steps distribution
    ax5 = axes[1, 1]
    ax5.hist(sarsa_steps[-1000:], bins=30, alpha=0.7, label='SARSA', density=True)
    ax5.hist(qlearning_steps[-1000:], bins=30, alpha=0.7, label='Q-Learning', density=True)
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Density')
    ax5.set_title('Final Steps Distribution (Last 1000 episodes)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance summary
    ax6 = axes[1, 2]
    
    # Calculate final performance metrics
    sarsa_final_reward = np.mean(sarsa_rewards[-1000:])
    qlearning_final_reward = np.mean(qlearning_rewards[-1000:])
    sarsa_final_steps = np.mean(sarsa_steps[-1000:])
    qlearning_final_steps = np.mean(qlearning_steps[-1000:])
    
    metrics = ['Avg Reward', 'Avg Steps']
    sarsa_values = [sarsa_final_reward, sarsa_final_steps]
    qlearning_values = [qlearning_final_reward, qlearning_final_steps]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, sarsa_values, width, label='SARSA', alpha=0.8)
    bars2 = ax6.bar(x + width/2, qlearning_values, width, label='Q-Learning', alpha=0.8)
    
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Value')
    ax6.set_title('Final Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def analyze_q_table(agent, agent_name="Agent"):
    """Analyze the learned Q-table"""
    
    print(f"\nAnalyzing {agent_name} Q-table...")
    
    # Convert defaultdict to regular dict for analysis
    q_dict = dict(agent.q_table)
    
    print(f"Number of states visited: {len(q_dict)}")
    
    # Analyze Q-values
    all_q_values = []
    for state_q_values in q_dict.values():
        all_q_values.extend(state_q_values)
    
    print(f"Q-value statistics:")
    print(f"  Mean: {np.mean(all_q_values):.3f}")
    print(f"  Std:  {np.std(all_q_values):.3f}")
    print(f"  Min:  {np.min(all_q_values):.3f}")
    print(f"  Max:  {np.max(all_q_values):.3f}")
    
    # Analyze action preferences
    action_counts = np.zeros(6)  # Taxi has 6 actions
    for state_q_values in q_dict.values():
        best_action = np.argmax(state_q_values)
        action_counts[best_action] += 1
    
    print(f"Action preferences (best action frequency):")
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        percentage = count / len(q_dict) * 100
        print(f"  {name}: {count} ({percentage:.1f}%)")

def visualize_taxi_policy(agent, sample_states=20):
    """Visualize policy for sample states"""
    
    env = gym.make('Taxi-v3')
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    
    print(f"\nPolicy Visualization (Sample of {sample_states} states):")
    print("-" * 60)
    
    # Get some sample states
    states_to_show = list(agent.q_table.keys())[:sample_states]
    
    for i, state in enumerate(states_to_show):
        # Decode state for better understanding
        env.env.s = state
        taxi_row, taxi_col, pass_loc, dest_idx = env.env.decode(state)
        
        # Get best action
        best_action = agent.get_policy(state)
        q_values = agent.q_table[state]
        
        print(f"State {i+1:2d}: Taxi=({taxi_row},{taxi_col}), "
              f"Pass={pass_loc}, Dest={dest_idx}")
        print(f"  Best Action: {action_names[best_action]}")
        print(f"  Q-values: {[f'{q:.2f}' for q in q_values]}")
        print()
    
    env.close()

def demonstrate_episode(agent, agent_name="Agent"):
    """Demonstrate a complete episode"""
    
    env = gym.make('Taxi-v3')
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    
    print(f"\nDemonstrating {agent_name} episode:")
    print("-" * 40)
    
    state = env.reset()
    total_reward = 0
    step = 0
    
    while step < 50:  # Limit steps for demonstration
        # Decode state
        taxi_row, taxi_col, pass_loc, dest_idx = env.env.decode(state)
        
        # Get action
        action = agent.get_policy(state)
        
        print(f"Step {step+1:2d}: Taxi=({taxi_row},{taxi_col}), "
              f"Pass={pass_loc}, Dest={dest_idx}")
        print(f"  Action: {action_names[action]}")
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        print(f"  Reward: {reward:3d}, Total: {total_reward:3d}")
        
        if done:
            print(f"  Episode completed in {step+1} steps!")
            break
        
        state = next_state
        step += 1
        print()
    
    env.close()
    return total_reward, step + 1

if __name__ == "__main__":
    print("SARSA vs Q-Learning on Taxi Environment")
    print("=" * 50)
    
    # Compare algorithms
    (sarsa_agent, qlearning_agent, 
     sarsa_rewards, qlearning_rewards,
     sarsa_test_rewards, qlearning_test_rewards) = compare_sarsa_qlearning()
    
    # Analyze Q-tables
    analyze_q_table(sarsa_agent, "SARSA")
    analyze_q_table(qlearning_agent, "Q-Learning")
    
    # Visualize policies
    visualize_taxi_policy(sarsa_agent, sample_states=10)
    
    # Demonstrate episodes
    print("\nDemonstration Episodes:")
    sarsa_demo_reward, sarsa_demo_steps = demonstrate_episode(sarsa_agent, "SARSA")
    qlearning_demo_reward, qlearning_demo_steps = demonstrate_episode(qlearning_agent, "Q-Learning")
    
    # Final comparison
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    
    print(f"SARSA:")
    print(f"  Training - Final avg reward: {np.mean(sarsa_rewards[-1000:]):.2f}")
    print(f"  Testing  - Avg reward: {np.mean(sarsa_test_rewards):.2f}")
    print(f"  Demo     - Reward: {sarsa_demo_reward}, Steps: {sarsa_demo_steps}")
    
    print(f"\nQ-Learning:")
    print(f"  Training - Final avg reward: {np.mean(qlearning_rewards[-1000:]):.2f}")
    print(f"  Testing  - Avg reward: {np.mean(qlearning_test_rewards):.2f}")
    print(f"  Demo     - Reward: {qlearning_demo_reward}, Steps: {qlearning_demo_steps}")
    
    # Determine winner
    sarsa_score = np.mean(sarsa_test_rewards)
    qlearning_score = np.mean(qlearning_test_rewards)
    
    if sarsa_score > qlearning_score:
        print(f"\n🏆 SARSA wins with {sarsa_score:.2f} vs {qlearning_score:.2f}")
    elif qlearning_score > sarsa_score:
        print(f"\n🏆 Q-Learning wins with {qlearning_score:.2f} vs {sarsa_score:.2f}")
    else:
        print(f"\n🤝 It's a tie! Both scored {sarsa_score:.2f}")