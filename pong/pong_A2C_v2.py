import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import gym

# Define distributions
Categorical = dist.Categorical
# Hyperparameters
lr = 1e-4
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
num_steps = 5
max_grad_norm = 0.5
max_episodes = 1000
batch_size = 4
print_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing function
def preprocess(observation):
    observation = observation[35:195]  # crop
    observation = observation[::2, ::2, 0]  # downsample by factor of 2 and keep only the first channel (R)
    observation[observation == 144] = 0  # erase background (type 1)
    observation[observation == 109] = 0  # erase background (type 2)
    observation[observation != 0] = 1  # set paddles and ball to 1
    observation = np.expand_dims(observation, axis=0)  # add batch dimension
    observation = torch.from_numpy(observation).float().to(device)
    return observation

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return F.softmax(self.actor(x), dim=1), self.critic(x)

    # A2C algorithm
    def train():
        envs = [gym.make("PongNoFrameskip-v4") for i in range(batch_size)]
        model = ActorCritic(envs[0].action_space.n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for episode in range(max_episodes):
            # Initialize batch data
            states = torch.zeros((num_steps, batch_size, 1, 64, 64)).to(device)
            actions = torch.zeros((num_steps, batch_size)).long().to(device)
            rewards = torch.zeros((num_steps, batch_size)).to(device)
            dones = torch.zeros((num_steps, batch_size)).to(device)
            values = torch.zeros((num_steps+1, batch_size)).to(device)
            log_probs = torch.zeros((num_steps, batch_size)).to(device)
            
            # Reset environments and get initial states
            for i, env in enumerate(envs):
                state = env.reset()
                state = preprocess(state)
                state = state.repeat(1, 4, 1, 1)
                states[0,i] = state
                values[0,i] = model(state)[1]
            
            # Collect trajectories
            for t in range(num_steps):
                with torch.no_grad():
                    policy, value = model(states[t])
                action = policy.multinomial(num_samples=1).squeeze(1)
                log_prob = F.log_softmax(policy, dim=1)[range(batch_size), action]
                reward = torch.zeros(batch_size).to(device)
                done = torch.zeros(batch_size).to(device)
                for i, env in enumerate(envs):
                    obs, r, d, _ = env.step(action[i].item())
                    obs = preprocess(obs)
                    states[t+1,i] = torch.cat((states[t,i,1:,:,:], obs), dim=0)
                    reward[i] = r
                    done[i] = d
                    if d:
                        obs = env.reset()
                        obs = preprocess(obs)
                        states[t+1,i] = obs.repeat(1, 4, 1, 1)
                        values[t+1,i] = model(obs)[1]
                actions[t] = action
                rewards[t] = reward
                dones[t] = done
                values[t+1] = value.squeeze()
                log_probs[t] = log_prob
            
            # Compute advantages and target values
            advantages = torch.zeros((num_steps, batch_size)).to(device)
            target_values = torch.zeros((num_steps, batch_size)).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - dones[-1]
                    next_values = model(states[-1])[1].squeeze()
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = values[t+1]
                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                advantages[t] = delta + gamma * value_loss_coef * next_non_terminal * advantages[t+1]
                target_values[t] = advantages[t] + values[t]
            
            # Flatten batch data
            states = states[:-1].view(-1, 1, 64, 64)
            actions = actions.view(-1)
            log_probs = log_probs.view(-1)
            advantages = advantages.view(-1)
            target_values = target_values.view(-1)
            
            # Compute loss and update model
            policy, value = model(states)
            dist = Categorical(F.softmax(policy, dim=1))
            entropy = dist.entropy().mean()
            policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
            value_loss = F.smooth_l1_loss(value.squeeze(), target_values.detach())
            loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Print training progress
            if episode % print_interval == 0:
                print("Episode: {0}\tLength: {1}\tTotal Reward: {2}\tLoss: {3:.3f}".format(
                    episode, t+1, rewards.sum().item(), loss.item()))
        
        # Close environments
        for env in envs:
            env.close()