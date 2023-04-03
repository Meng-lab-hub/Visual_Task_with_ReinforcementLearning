
# Import the necessary software library --------------------------------------------------------------------------

import os
import torch
import gym
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

import torchvision.transforms as transforms
import json

# from utils import test_policy_network, seed_everything, plot_stats
from parallel_env import ParallelEnv, ParallelWrapper



# Classes Definition ---------------------------------------------------------------------------------------------
class Environment():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
    
    def seed_everything(self, seed = 42):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def create_env(self):
        self.seed_everything()
        return self.env

class PreprocessEnv(ParallelWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        state = self.venv.reset()
        return torch.from_numpy(state).float()

    def step_async(self, actions):
        # actions = actions.squeeze().cpu().numpy()
        actions = actions.squeeze().to(device=torch.device("cuda")).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        next_state, reward, done, info = self.venv.step_wait()
        next_state = torch.from_numpy(next_state).float()
        reward = torch.tensor(reward).unsqueeze(1).float()
        done = torch.tensor(done).unsqueeze(1)
        return next_state, reward, done, info
    
class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 512)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        # change F to nn?
        
        x = x.view(-1, 64*4*4)
        
        x = F.relu(self.fc1(x))
        return x
    
class Actor(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x
    
class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class ActorCritic():
    def __init__(self, actor, critic, feature, device, num_envs, num_frames, alpha=7e-4, gamma=0.99):
        self.actor = actor
        self.critic = critic
        self.feature = feature
        # self.alpha = alpha
        self.gamma = gamma
        self.actor_optim = AdamW(self.actor.parameters(), lr=alpha)
        self.critic_optim = AdamW(self.critic.parameters(), lr=alpha)
        self.feature_optim = AdamW(self.feature.parameters(), lr=alpha)
        self.stats = {'Actor Loss': [], 'Critic Loss': [],'Returns': []}
        self.best_return = -999
        self.device = device
        self.num_envs = num_envs
        self.num_frames = num_frames
        self.stacked_frames = torch.zeros((num_envs, num_frames, 64, 64))
        # self.actor_actions = []


    def preprocess_observation(self, obs_batch):
        # Crop the score and border region
        obs_batch = obs_batch[:, 35:195, :, :]

        # Convert to grayscale
        obs_batch = torch.mean(obs_batch, dim=3, keepdim=True)
    
        # Resize to 64x64
        transform = transforms.Resize((64, 64))
        obs_batch = torch.stack([transform(obs.permute(2, 0, 1)) for obs in obs_batch])
        
        # Convert to float and rescale to range [0, 1]
        obs_batch = obs_batch / 255.0
        
        return obs_batch
    
    def save_model(self, path="/media/data/meng/best_model.pth.tar"):
        torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'feature_state_dict': self.feature.state_dict(),
                    'actor_optim_state_dict': self.actor_optim.state_dict(),
                    'critic_optim_state_dict': self.critic_optim.state_dict(),
                    'feature_optim_state_dict': self.feature_optim.state_dict(),
                }, path)

    def save_stats(self, filename='/media/data/meng/stats.txt'):
        with open(filename, 'w') as f:
            json.dump(self.stats, f)

    def save_action_decision(self, filename='/media/data/meng/actor_actions.txt'):
        with open(filename, 'w') as f:
            flat_actions = [a.item() for sublist in self.actor_actions for a in sublist]
            json.dump(flat_actions, f)
        
    def save_input_states(self, filename='/media/data/meng/input_states.txt'):
        with open(filename, 'w') as f:
            input_states_mod = torch.cat(self.input_states)
            input_states_mod = input_states_mod.cpu().flatten().numpy()
            input_states_mod = input_states_mod.tolist()
            json.dump(input_states_mod, f)

    def create_frames_batch(self, env, state):
        for i in range(self.num_frames):
            tmp = state.squeeze(dim=1)
            self.stacked_frames[:, i, :, :] = tmp.clone()

    def add_frame(self, state):
        tmp = torch.cat((self.stacked_frames[:, 1:, :, :], state), dim=1)
        self.stacked_frames = tmp.clone()
        

    def train(self, env, episodes):

        self.stacked_frames = self.stacked_frames.to(self.device)
        for episode in tqdm(range(1, episodes + 1)):
            state = env.reset()
            state = self.preprocess_observation(state)
            state = state.to(self.device)

            # for ploting a histogram of input state----#
            # self.input_states.append(state)           #
            # ------------------------------------------#

            done_b = torch.zeros((env.num_envs, 1), dtype=torch.bool)
            done_b = done_b.to(self.device)

            ep_return = torch.zeros((env.num_envs, 1))
            ep_return = ep_return.to(self.device)

            I = 1.

            while not done_b.all():
                
                self.critic_optim.zero_grad()
                self.actor_optim.zero_grad()
                self.feature_optim.zero_grad()
                
                featured = self.feature(self.stacked_frames)
                action_probs = self.actor(featured)
                
                # action = self.actor(featured).multinomial(1).detach()

                action = torch.multinomial(action_probs, 1).squeeze().detach()

                # save actions for ploting histogram---#
                # self.actor_actions.append(action)    #
                # -------------------------------------#

                next_state, reward, done, _ = env.step(action)
                next_state = self.preprocess_observation(next_state)

                # in case use GPU to compute--------------------#
                next_state = next_state.to(self.device)         #
                reward = reward.to(self.device)                 #
                done = done.to(self.device)                     #
                # ----------------------------------------------#
                self.add_frame(next_state)

                # try increase a reward----#
                # reward = reward * 10     #
                # -------------------------#
                
                value = self.critic(featured)

                next_featured = self.feature(self.stacked_frames)

                target = reward + ~done * self.gamma * self.critic(next_featured).detach()
                critic_loss = F.mse_loss(value, target)
                
                advantage = (target - value).detach()

                log_probs = torch.log(action_probs + 1e-6)
                action = action.view(-1, 1)
                action_log_prob = log_probs.gather(1, action)
                entropy = - torch.sum(action_probs * log_probs, dim=-1, keepdim=True)

                actor_loss = - I * action_log_prob * advantage - 0.01 * entropy
                actor_loss = actor_loss.mean()
                
                total_loss = actor_loss + critic_loss
                
                total_loss.backward()

                self.feature_optim.step()
                self.critic_optim.step()
                self.actor_optim.step()

                ep_return += reward
                done_b |= done
                state = next_state
                I = I * self.gamma

            # A piece of code use to save a model when we got a highest return---------#
            current_return = ep_return.mean().item()                                   #
            if self.best_return < current_return:                                      #
                self.best_return = current_return                                      #
                self.save_model()                                                      #
            # -------------------------------------------------------------------------#

            self.stats['Actor Loss'].append(actor_loss.item())
            self.stats['Critic Loss'].append(critic_loss.item())
            self.stats['Returns'].append(ep_return.mean().item())

        

# Main function ---------------------------------------------------------------------------------------------------
def main():

    # define parameter
    env_name = 'PongNoFrameskip-v4' 
    num_envs = os.cpu_count()
    num_frames = 4
    episodes = 5000

    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # creating parallel environment
    envs = [lambda: Environment(env_name).create_env() for _ in range(num_envs)]
    envs = ParallelEnv(envs)
    envs = PreprocessEnv(envs)


    # instanciating our agent
    feature = FeatureExtractor().to(device)
    actor = Actor(envs.action_space.n).to(device)
    critic = Critic().to(device)
    agent = ActorCritic(actor, critic, feature, device, num_envs, num_frames)

    # train out agent
    agent.train(envs, episodes)

    print(f"Trained successfully")

    # save infos
    agent.save_stats()
    # agent.save_action_decision()
    # agent.save_input_states()

    print(f"Saved successfully")

if __name__ == "__main__" : main()