import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MultiAgentGridWorld:
    """Multi-agent grid world environment"""
    
    def __init__(self, width=8, height=8, num_agents=3, num_goals=3):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_goals = num_goals
        
        # Environment setup
        self.obstacles = [(2, 2), (2, 3), (3, 2), (5, 5), (5, 6), (6, 5)]
        self.goals = [(1, 1), (6, 1), (1, 6)][:num_goals]
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        # Random agent positions (avoiding obstacles and goals)
        self.agent_positions = []
        for _ in range(self.num_agents):
            while True:
                pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
                if (pos not in self.obstacles and 
                    pos not in self.goals and 
                    pos not in self.agent_positions):
                    self.agent_positions.append(pos)
                    break
        
        # Track which goals are collected
        self.goals_collected = [False] * len(self.goals)
        self.step_count = 0
        
        return self.get_observations()
    
    def get_observations(self):
        """Get observations for all agents"""
        observations = []
        
        for i, agent_pos in enumerate(self.agent_positions):
            obs = np.zeros((self.height, self.width, 4))  # 4 channels
            
            # Channel 0: Agent positions
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:  # Don't include self
                    obs[other_pos[0], other_pos[1], 0] = 1
            
            # Channel 1: Goals (uncollected)
            for j, goal_pos in enumerate(self.goals):
                if not self.goals_collected[j]:
                    obs[goal_pos[0], goal_pos[1], 1] = 1
            
            # Channel 2: Obstacles
            for obs_pos in self.obstacles:
                obs[obs_pos[0], obs_pos[1], 2] = 1
            
            # Channel 3: Self position
            obs[agent_pos[0], agent_pos[1], 3] = 1
            
            # Flatten observation
            observations.append(obs.flatten())
        
        return observations
    
    def step(self, actions):
        """Take actions for all agents"""
        # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        
        new_positions = []
        rewards = []
        
        # Calculate new positions
        for i, action in enumerate(actions):
            current_pos = self.agent_positions[i]
            move = moves[action]
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            
            # Check boundaries
            if (0 <= new_pos[0] < self.height and 
                0 <= new_pos[1] < self.width and 
                new_pos not in self.obstacles):
                new_positions.append(new_pos)
            else:
                new_positions.append(current_pos)  # Stay in place
        
        # Handle collisions (agents can't occupy same cell)
        final_positions = []
        for i, new_pos in enumerate(new_positions):
            # Check if position conflicts with other agents
            conflict = False
            for j, other_new_pos in enumerate(new_positions):
                if i != j and new_pos == other_new_pos:
                    conflict = True
                    break
            
            if conflict:
                final_positions.append(self.agent_positions[i])  # Stay in original position
            else:
                final_positions.append(new_pos)
        
        # Calculate rewards
        for i, agent_pos in enumerate(final_positions):
            reward = -0.01  # Small step penalty
            
            # Check if agent reached a goal
            for j, goal_pos in enumerate(self.goals):
                if agent_pos == goal_pos and not self.goals_collected[j]:
                    reward += 10  # Goal reward
                    self.goals_collected[j] = True
                    break
            
            # Penalty for collision attempt
            if final_positions[i] == self.agent_positions[i] and new_positions[i] != self.agent_positions[i]:
                reward -= 0.1
            
            rewards.append(reward)
        
        # Update positions
        self.agent_positions = final_positions
        self.step_count += 1
        
        # Check if done (all goals collected or max steps)
        done = all(self.goals_collected) or self.step_count >= 200
        
        return self.get_observations(), rewards, done
    
    def render(self):
        """Render the environment"""
        grid = np.zeros((self.height, self.width))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goals
        for i, goal in enumerate(self.goals):
            if not self.goals_collected[i]:
                grid[goal] = 2
        
        # Mark agents
        for i, agent_pos in enumerate(self.agent_positions):
            grid[agent_pos] = i + 3
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='tab10')
        plt.title(f'Multi-Agent Grid World (Step {self.step_count})')
        
        # Add text annotations
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.obstacles:
                    plt.text(j, i, 'X', ha='center', va='center', fontsize=12, color='white')
                elif (i, j) in self.goals and not self.goals_collected[self.goals.index((i, j))]:
                    plt.text(j, i, 'G', ha='center', va='center', fontsize=12, color='white')
                elif (i, j) in self.agent_positions:
                    agent_id = self.agent_positions.index((i, j))
                    plt.text(j, i, str(agent_id), ha='center', va='center', fontsize=12, color='white')
        
        plt.show()

class AgentNetwork(nn.Module):
    """Neural network for individual agent"""
    
    def __init__(self, obs_size, action_size, hidden_size=128):
        super(AgentNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        policy_logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return policy_logits, value
    
    def get_action_and_value(self, obs):
        """Get action and value for given observation"""
        policy_logits, value = self.forward(obs)
        
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()

class MultiAgentSystem:
    """Multi-agent reinforcement learning system"""
    
    def __init__(self, env, lr=1e-3, gamma=0.99):
        self.env = env
        self.num_agents = env.num_agents
        self.gamma = gamma
        
        # Get observation and action sizes
        obs = env.get_observations()
        self.obs_size = len(obs[0])
        self.action_size = 5  # 4 directions + stay
        
        # Create networks for each agent
        self.agents = []
        self.optimizers = []
        
        for i in range(self.num_agents):
            agent = AgentNetwork(self.obs_size, self.action_size)
            optimizer = optim.Adam(agent.parameters(), lr=lr)
            
            self.agents.append(agent)
            self.optimizers.append(optimizer)
        
        # Storage for trajectories
        self.reset_trajectories()
    
    def reset_trajectories(self):
        """Reset trajectory storage for all agents"""
        self.trajectories = []
        for _ in range(self.num_agents):
            self.trajectories.append({
                'observations': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'dones': []
            })
    
    def get_actions(self, observations):
        """Get actions for all agents"""
        actions = []
        log_probs = []
        values = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = self.agents[i].get_action_and_value(obs_tensor)
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        
        return actions, log_probs, values
    
    def store_transitions(self, observations, actions, rewards, log_probs, values, done):
        """Store transitions for all agents"""
        for i in range(self.num_agents):
            self.trajectories[i]['observations'].append(observations[i])
            self.trajectories[i]['actions'].append(actions[i])
            self.trajectories[i]['rewards'].append(rewards[i])
            self.trajectories[i]['log_probs'].append(log_probs[i].item())
            self.trajectories[i]['values'].append(values[i].item())
            self.trajectories[i]['dones'].append(done)
    
    def compute_returns_and_advantages(self, agent_id, next_value=0):
        """Compute returns and advantages for specific agent"""
        traj = self.trajectories[agent_id]
        
        returns = []
        advantages = []
        
        # Add next value for bootstrapping
        values = traj['values'] + [next_value]
        
        # Compute returns
        G = next_value
        for i in reversed(range(len(traj['rewards']))):
            G = traj['rewards'][i] + self.gamma * G * (1 - traj['dones'][i])
            returns.insert(0, G)
        
        # Compute advantages
        for i in range(len(traj['rewards'])):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        
        return returns, advantages
    
    def update_agent(self, agent_id, next_value=0):
        """Update specific agent"""
        traj = self.trajectories[agent_id]
        
        if len(traj['observations']) == 0:
            return 0, 0
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(agent_id, next_value)
        
        # Convert to tensors
        observations = torch.FloatTensor(traj['observations'])
        actions = torch.LongTensor(traj['actions'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(traj['log_probs'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        policy_logits, values = self.agents[agent_id](observations)
        
        # Actor loss
        dist = Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        entropy_bonus = 0.01 * entropy
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - entropy_bonus
        
        # Update
        self.optimizers[agent_id].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].parameters(), 0.5)
        self.optimizers[agent_id].step()
        
        return actor_loss.item(), critic_loss.item()
    
    def update_all_agents(self, next_observations=None):
        """Update all agents"""
        actor_losses = []
        critic_losses = []
        
        # Get next values for bootstrapping
        next_values = [0] * self.num_agents
        if next_observations is not None:
            for i, next_obs in enumerate(next_observations):
                next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)
                with torch.no_grad():
                    _, next_value = self.agents[i](next_obs_tensor)
                    next_values[i] = next_value.item()
        
        # Update each agent
        for i in range(self.num_agents):
            actor_loss, critic_loss = self.update_agent(i, next_values[i])
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        # Reset trajectories
        self.reset_trajectories()
        
        return actor_losses, critic_losses

def train_multi_agent_system(episodes=2000, max_steps=200, update_frequency=50):
    """Train multi-agent system"""
    
    # Create environment and system
    env = MultiAgentGridWorld(width=8, height=8, num_agents=3, num_goals=3)
    mas = MultiAgentSystem(env)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    goals_collected_per_episode = []
    actor_losses = [[] for _ in range(mas.num_agents)]
    critic_losses = [[] for _ in range(mas.num_agents)]
    
    print(f"Training Multi-Agent System")
    print(f"Environment: {env.width}x{env.height} grid")
    print(f"Agents: {env.num_agents}, Goals: {env.num_goals}")
    print("-" * 50)
    
    for episode in range(episodes):
        observations = env.reset()
        episode_reward = [0] * mas.num_agents
        episode_length = 0
        
        for step in range(max_steps):
            # Get actions
            actions, log_probs, values = mas.get_actions(observations)
            
            # Take step
            next_observations, rewards, done = env.step(actions)
            
            # Store transitions
            mas.store_transitions(observations, actions, rewards, log_probs, values, done)
            
            # Update episode metrics
            for i in range(mas.num_agents):
                episode_reward[i] += rewards[i]
            episode_length += 1
            
            # Update agents periodically
            if len(mas.trajectories[0]['observations']) >= update_frequency or done:
                if not done:
                    a_losses, c_losses = mas.update_all_agents(next_observations)
                else:
                    a_losses, c_losses = mas.update_all_agents()
                
                for i in range(mas.num_agents):
                    actor_losses[i].append(a_losses[i])
                    critic_losses[i].append(c_losses[i])
            
            observations = next_observations
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        goals_collected_per_episode.append(sum(env.goals_collected))
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean([sum(rewards) for rewards in episode_rewards[-100:]])
            avg_goals = np.mean(goals_collected_per_episode[-100:])
            print(f"Episode {episode:4d} | Avg Total Reward: {avg_reward:6.2f} | "
                  f"Avg Goals: {avg_goals:.2f} | Length: {episode_length:3d}")
        
        # Check if solved (consistently collecting all goals)
        if len(goals_collected_per_episode) >= 100 and np.mean(goals_collected_per_episode[-100:]) >= 2.8:
            print(f"\nEnvironment solved in {episode} episodes!")
            print(f"Average goals collected: {np.mean(goals_collected_per_episode[-100:]):.2f}")
            break
    
    return mas, episode_rewards, episode_lengths, goals_collected_per_episode, actor_losses, critic_losses

def test_multi_agent_system(mas, episodes=10, render=False):
    """Test trained multi-agent system"""
    env = MultiAgentGridWorld(width=8, height=8, num_agents=3, num_goals=3)
    test_rewards = []
    test_goals = []
    test_lengths = []
    
    print(f"\nTesting multi-agent system for {episodes} episodes...")
    
    for episode in range(episodes):
        observations = env.reset()
        episode_reward = [0] * mas.num_agents
        episode_length = 0
        
        if render and episode < 3:
            print(f"\nTest Episode {episode + 1}:")
            env.render()
        
        for step in range(200):
            # Get actions (deterministic for testing)
            actions = []
            for i, obs in enumerate(observations):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, _ = mas.agents[i](obs_tensor)
                    action = policy_logits.argmax().item()
                actions.append(action)
            
            observations, rewards, done = env.step(actions)
            
            for i in range(mas.num_agents):
                episode_reward[i] += rewards[i]
            episode_length += 1
            
            if render and episode < 3 and step % 20 == 0:
                env.render()
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_goals.append(sum(env.goals_collected))
        test_lengths.append(episode_length)
        
        print(f"Test Episode {episode + 1}: Total Reward = {sum(episode_reward):.2f}, "
              f"Goals = {sum(env.goals_collected)}, Length = {episode_length}")
    
    avg_total_reward = np.mean([sum(rewards) for rewards in test_rewards])
    avg_goals = np.mean(test_goals)
    avg_length = np.mean(test_lengths)
    
    print(f"\nTest Results:")
    print(f"Average total reward: {avg_total_reward:.2f}")
    print(f"Average goals collected: {avg_goals:.2f}")
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Success rate (all goals): {np.mean([g == 3 for g in test_goals])*100:.1f}%")
    
    return test_rewards, test_goals, test_lengths

def plot_multi_agent_results(episode_rewards, episode_lengths, goals_collected, 
                           actor_losses, critic_losses):
    """Plot multi-agent training results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Total rewards over episodes
    ax1 = axes[0, 0]
    total_rewards = [sum(rewards) for rewards in episode_rewards]
    ax1.plot(total_rewards, alpha=0.6)
    
    # Moving average
    window_size = 100
    if len(total_rewards) >= window_size:
        moving_avg = [np.mean(total_rewards[max(0, i-window_size+1):i+1]) 
                     for i in range(len(total_rewards))]
        ax1.plot(moving_avg, color='red', linewidth=2, 
                label=f'Moving Average ({window_size})')
        ax1.legend()
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Rewards over Episodes')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Goals collected over episodes
    ax2 = axes[0, 1]
    ax2.plot(goals_collected, alpha=0.6)
    
    if len(goals_collected) >= window_size:
        goals_ma = [np.mean(goals_collected[max(0, i-window_size+1):i+1]) 
                   for i in range(len(goals_collected))]
        ax2.plot(goals_ma, color='red', linewidth=2, 
                label=f'Moving Average ({window_size})')
        ax2.legend()
    
    ax2.axhline(y=3, color='green', linestyle='--', label='All Goals')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Goals Collected')
    ax2.set_title('Goals Collected over Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode lengths
    ax3 = axes[0, 2]
    ax3.plot(episode_lengths, alpha=0.6)
    
    if len(episode_lengths) >= window_size:
        length_ma = [np.mean(episode_lengths[max(0, i-window_size+1):i+1]) 
                    for i in range(len(episode_lengths))]
        ax3.plot(length_ma, color='red', linewidth=2, 
                label=f'Moving Average ({window_size})')
        ax3.legend()
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Lengths')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Individual agent rewards
    ax4 = axes[1, 0]
    for i in range(len(episode_rewards[0])):
        agent_rewards = [rewards[i] for rewards in episode_rewards]
        if len(agent_rewards) >= window_size:
            agent_ma = [np.mean(agent_rewards[max(0, j-window_size+1):j+1]) 
                       for j in range(len(agent_rewards))]
            ax4.plot(agent_ma, label=f'Agent {i}', alpha=0.8)
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Individual Agent Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Actor losses
    ax5 = axes[1, 1]
    for i, losses in enumerate(actor_losses):
        if losses:
            ax5.plot(losses, label=f'Agent {i}', alpha=0.8)
    
    ax5.set_xlabel('Update')
    ax5.set_ylabel('Actor Loss')
    ax5.set_title('Actor Losses')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance distribution
    ax6 = axes[1, 2]
    final_total_rewards = total_rewards[-100:] if len(total_rewards) >= 100 else total_rewards
    final_goals = goals_collected[-100:] if len(goals_collected) >= 100 else goals_collected
    
    ax6.hist(final_total_rewards, bins=20, alpha=0.7, label='Total Rewards', density=True)
    ax6_twin = ax6.twinx()
    ax6_twin.hist(final_goals, bins=4, alpha=0.7, label='Goals', color='orange', density=True)
    
    ax6.set_xlabel('Value')
    ax6.set_ylabel('Density (Rewards)', color='blue')
    ax6_twin.set_ylabel('Density (Goals)', color='orange')
    ax6.set_title('Final Performance Distribution')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_cooperation(mas, episodes=50):
    """Analyze cooperation between agents"""
    
    env = MultiAgentGridWorld(width=8, height=8, num_agents=3, num_goals=3)
    
    cooperation_metrics = {
        'goal_sharing': [],  # How evenly goals are distributed
        'collision_rate': [],  # Rate of collision attempts
        'coordination': []  # How well agents coordinate
    }
    
    print("\nAnalyzing cooperation between agents...")
    
    for episode in range(episodes):
        observations = env.reset()
        goals_per_agent = [0] * mas.num_agents
        collision_attempts = 0
        total_steps = 0
        
        for step in range(200):
            # Get actions
            actions = []
            for i, obs in enumerate(observations):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, _ = mas.agents[i](obs_tensor)
                    action = policy_logits.argmax().item()
                actions.append(action)
            
            # Check for potential collisions
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
            new_positions = []
            for i, action in enumerate(actions):
                current_pos = env.agent_positions[i]
                move = moves[action]
                new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
                new_positions.append(new_pos)
            
            # Count collision attempts
            for i in range(len(new_positions)):
                for j in range(i+1, len(new_positions)):
                    if new_positions[i] == new_positions[j]:
                        collision_attempts += 1
            
            # Take step
            old_goals = env.goals_collected.copy()
            observations, rewards, done = env.step(actions)
            
            # Track which agent collected which goal
            for i, goal_collected in enumerate(env.goals_collected):
                if goal_collected and not old_goals[i]:
                    # Find which agent is at this goal
                    goal_pos = env.goals[i]
                    for j, agent_pos in enumerate(env.agent_positions):
                        if agent_pos == goal_pos:
                            goals_per_agent[j] += 1
                            break
            
            total_steps += 1
            
            if done:
                break
        
        # Calculate metrics
        if sum(goals_per_agent) > 0:
            goal_variance = np.var(goals_per_agent)
            cooperation_metrics['goal_sharing'].append(goal_variance)
        
        collision_rate = collision_attempts / total_steps if total_steps > 0 else 0
        cooperation_metrics['collision_rate'].append(collision_rate)
        
        # Coordination metric (inverse of episode length for successful episodes)
        if sum(env.goals_collected) == len(env.goals):
            coordination = 1.0 / total_steps
        else:
            coordination = 0
        cooperation_metrics['coordination'].append(coordination)
    
    # Plot cooperation analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Goal sharing (lower variance = better sharing)
    ax1 = axes[0]
    ax1.hist(cooperation_metrics['goal_sharing'], bins=15, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Goal Variance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Goal Sharing Distribution\n(Lower = Better Cooperation)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Collision rate
    ax2 = axes[1]
    ax2.hist(cooperation_metrics['collision_rate'], bins=15, alpha=0.7, 
             edgecolor='black', color='orange')
    ax2.set_xlabel('Collision Rate')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Collision Rate Distribution\n(Lower = Better Coordination)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coordination efficiency
    ax3 = axes[2]
    coordination_nonzero = [c for c in cooperation_metrics['coordination'] if c > 0]
    if coordination_nonzero:
        ax3.hist(coordination_nonzero, bins=15, alpha=0.7, 
                edgecolor='black', color='green')
    ax3.set_xlabel('Coordination Efficiency')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Coordination Efficiency\n(Higher = Better)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Cooperation Analysis Results:")
    print(f"Average goal variance: {np.mean(cooperation_metrics['goal_sharing']):.3f}")
    print(f"Average collision rate: {np.mean(cooperation_metrics['collision_rate']):.3f}")
    print(f"Success rate: {np.mean([c > 0 for c in cooperation_metrics['coordination']])*100:.1f}%")
    print(f"Average coordination efficiency: {np.mean(coordination_nonzero) if coordination_nonzero else 0:.3f}")
    
    return cooperation_metrics

def compare_independent_vs_shared():
    """Compare independent learning vs shared policy"""
    
    print("Comparing Independent Learning vs Shared Policy...")
    
    # Train independent agents
    print("\nTraining with independent agents...")
    env1 = MultiAgentGridWorld(width=8, height=8, num_agents=3, num_goals=3)
    mas_independent = MultiAgentSystem(env1)
    
    # Train for fewer episodes for comparison
    (mas_independent, ind_rewards, ind_lengths, ind_goals, 
     ind_actor_losses, ind_critic_losses) = train_multi_agent_system(episodes=1000)
    
    # Test independent agents
    ind_test_rewards, ind_test_goals, ind_test_lengths = test_multi_agent_system(
        mas_independent, episodes=20)
    
    print(f"\nIndependent Learning Results:")
    print(f"Final avg total reward: {np.mean([sum(r) for r in ind_rewards[-100:]]):.2f}")
    print(f"Final avg goals: {np.mean(ind_goals[-100:]):.2f}")
    print(f"Test avg goals: {np.mean(ind_test_goals):.2f}")
    
    return mas_independent, ind_test_rewards, ind_test_goals

if __name__ == "__main__":
    print("Multi-Agent Reinforcement Learning")
    print("=" * 50)
    
    # Train multi-agent system
    (mas, episode_rewards, episode_lengths, goals_collected, 
     actor_losses, critic_losses) = train_multi_agent_system(episodes=1500)
    
    # Plot training results
    plot_multi_agent_results(episode_rewards, episode_lengths, goals_collected,
                            actor_losses, critic_losses)
    
    # Test the trained system
    test_rewards, test_goals, test_lengths = test_multi_agent_system(mas, episodes=10, render=True)
    
    # Analyze cooperation
    cooperation_metrics = analyze_cooperation(mas, episodes=100)
    
    # Compare with different approaches
    mas_independent, ind_test_rewards, ind_test_goals = compare_independent_vs_shared()
    
    # Save the trained models
    for i, agent in enumerate(mas.agents):
        torch.save(agent.state_dict(), f'multi_agent_{i}.pth')
    
    print("\nModels saved!")
    print(f"Training completed!")
    
    # Final statistics
    final_total_rewards = [sum(rewards) for rewards in episode_rewards[-100:]]
    final_goals = goals_collected[-100:]
    
    print(f"Final average total reward: {np.mean(final_total_rewards):.2f}")
    print(f"Final average goals collected: {np.mean(final_goals):.2f}")
    print(f"Test average goals collected: {np.mean(test_goals):.2f}")
    
    # Check if solved
    if np.mean(final_goals) >= 2.8:
        print("Multi-agent environment SOLVED! 🎉")
    else:
        print("Significant progress achieved in multi-agent coordination.")
    
    # Cooperation summary
    print(f"\nCooperation Summary:")
    print(f"Goal sharing variance: {np.mean(cooperation_metrics['goal_sharing']):.3f}")
    print(f"Collision rate: {np.mean(cooperation_metrics['collision_rate']):.3f}")
    success_rate = np.mean([c > 0 for c in cooperation_metrics['coordination']]) * 100
    print(f"Coordination success rate: {success_rate:.1f}%")