"""
Implementation inspired from Proximal Policy Optimization (PPO) algorithm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gymnasium as gym

# Backbone network shared by actor and critic
class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Actor-critic model
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        # logits for actions
        action_logits = self.actor(state)
        # value for state
        value = self.critic(state)
        return action_logits, value

def create_agent(hidden_dim, dropout, env):
    # Create actor and critic networks based on the environment's observation and action spaces
    input_dim = env.observation_space.shape[0]
    actor_output_dim = env.action_space.n
    # critic_output_dim = 1  # Single value for state value
    critic_output_dim = 1
    actor = BackboneNetwork(input_dim, hidden_dim, actor_output_dim, dropout)
    critic = BackboneNetwork(input_dim, hidden_dim, critic_output_dim, dropout)
    return ActorCritic(actor, critic)

def calculate_returns(rewards, gamma):
    # Calculate cumulative discounted returns
    returns = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + cumulative * gamma
        returns.insert(0, cumulative)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# difference from baselie advantage
def calculate_advantages(returns, values):
    # Calculate advantages using Generalized Advantage Estimation (GAE)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def calculate_surrogate_loss(old_log_probs, new_log_probs, epsilon, advantages):
    # Calculate surrogate loss for PPO
    ratios = (new_log_probs - old_log_probs).exp()
    # Calculate the surrogate loss using clipped objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, min=1.0 - epsilon, max=1.0 + epsilon) * advantages
    return torch.min(surr1, surr2)

def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, values):
    # Calculate policy and value losses
    policy_loss = -(surrogate_loss + entropy_coefficient * entropy).sum()
    value_loss = F.smooth_l1_loss(values, returns).sum()
    return policy_loss, value_loss

def forward_pass(env, agent, optimizer, gamma):
    states, actions, log_probs, values, rewards = [], [], [], [], []
    done = False
    state, _ = env.reset()
    episode_reward = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_logits, value = agent(state_tensor)
        action_prob = F.softmax(action_logits, dim=-1)
        # Sample action from the action distribution
        dist = torch.distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Store the state, action, log probability, and value
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        episode_reward += reward
        state = next_state
    # Convert lists to tensors
    states = torch.cat(states)
    actions = torch.cat(actions)
    log_probs = torch.cat(log_probs)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, gamma)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, log_probs, advantages, returns

def update_policy(agent, states, actions, old_log_probs, advantages, returns,
                  optimizer, ppo_steps, epsilon, entropy_coefficient):
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    old_log_probs = old_log_probs.detach()
    actions = actions.detach()
    advantages = advantages.detach()           # FIX: Detach
    returns = returns.detach()                 # FIX: Detach
    states = states.detach()
    old_log_probs = old_log_probs.detach()
    actions = actions.detach()
    data = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(ppo_steps):
        for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in loader:
            action_logits, value = agent(batch_states)
            value = value.squeeze(-1)
            action_prob = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            entropy = dist.entropy()
            new_log_probs = dist.log_prob(batch_actions)
            surrogate_loss = calculate_surrogate_loss(
                batch_old_log_probs, new_log_probs, epsilon, batch_advantages
            )
            policy_loss, value_loss = calculate_losses(
                surrogate_loss, entropy, entropy_coefficient, batch_returns, value
            )
            optimizer.zero_grad()
            total_loss = policy_loss + value_loss
            total_loss.backward()
            # value_loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, agent):
    agent.eval()
    done = False
    total_reward = 0
    state, _ = env.reset()
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = agent(state_tensor)
            action_prob = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        state = next_state
    agent.train()
    return total_reward

def run_ppo():
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3

    env_train = gym.make('CartPole-v1')
    env_test = gym.make('CartPole-v1')
    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT, env_train)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    train_rewards, test_rewards, policy_losses, value_losses = [], [], [], []

    for episode in range(1, MAX_EPISODES + 1):
        train_reward, states, actions, log_probs, advantages, returns = forward_pass(
            env_train, agent, optimizer, DISCOUNT_FACTOR)
        policy_loss, value_loss = update_policy(
            agent, states, actions, log_probs, advantages, returns, optimizer, PPO_STEPS,
            EPSILON, ENTROPY_COEFFICIENT)
        test_reward = evaluate(env_test, agent)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))

        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | '
                  f'Mean Train Rewards: {mean_train_rewards:3.1f} | '
                  f'Mean Test Rewards: {mean_test_rewards:3.1f} | '
                  f'Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} | '
                  f'Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

if __name__ == "__main__":
    run_ppo()
