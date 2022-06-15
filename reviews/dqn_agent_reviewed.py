import numpy as np
import random
from collections import namedtuple, deque
import gym

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
DDQN = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env_conf, seed):
        state_size = env_conf["state_size"]
        self.action_size = env_conf["action_size"]
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, self.action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)


    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)        


    def interact_with_environment(self, env, state, eps, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # compute the Q-value for all actions at once in a particular state
        # input (S) -> output (Q(S, a1), Q(S, a2), Q(S, a3) ... Q(S, an))
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action =  random.choice(np.arange(self.action_size))
        next_state, reward, done, _ = env.step(action)
        return action, reward, next_state, done


    def learn(self):
        # - get and sample data from the replay buffer
        # - perform inference using the TARGET network (DQN as well) with Sₜ₊₁ as input
        # - Get the MAX Q value predicted: Q_target_next
        # - Compute the first part of the error term: Q_target = R + (gamma * Q_targets_next) * (1 - done) this last term is because Q_target = R is the episode is over
        # - perform inference using the LOCAL network (DQN as well) with S as input
        # - Get the values predicted for the actions selected A during interractions : Q_local
        # - Compute the second part using the LOCAL network: Q_expected
        # - Get the loss using Q_expected and Q_target as input (mse loss)
        # - Minimize the loss : self.optimizer.zero_grad(), loss.backward(), self.optimizer.step()
        # - Finally update the Target network

        states, actions, rewards, next_states, dones = self.memory.sample()
        
        if DDQN:
            argmax_q_next = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(-1)
            q_next = self.qnetwork_target(next_states).gather(1, argmax_q_next)
            Q_targets = rewards + (GAMMA * q_next * (1 - dones))
        else:
            # forward prop to get actions-values + take the max Q(Sₜ₊₁,a) of the next state
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # use bellman equation to compute optimal action-value function of the current state
            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from : 𝐖
            target_model (PyTorch model): weights will be copied to : our fixed Q-target 𝐖⁻
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # give me batch_size "randomly" selected <S, A, Rₜ₊₁, Sₜ₊₁> in a list
        experiences = random.sample(self.memory, k=self.batch_size)

        # stack all the states together and convert them to a tensor
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)  # (batch_size x 5)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    environment_configuration = {"state_size": 8, "action_size": 4}
    
    n_episodes = 2000
    max_time_steps = 1000
    learn_every = 5
    score_episodes = []
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    agent = Agent(environment_configuration, seed=0)
    eps = eps_start

    for episode in range(n_episodes+1):
        state = env.reset(seed=0)
        sum_rewards = 0

        for t in range(max_time_steps):
            action, reward, next_state, done = agent.interact_with_environment(env, state, eps, device)
            agent.store_experience(state, action, reward, next_state, done)

            if t % learn_every == 0 and len(agent.memory) >= BATCH_SIZE:
                agent.learn()  # sample and learn

            state = next_state
            sum_rewards += reward
            if done:
                break
        
        score_episodes.append(sum_rewards)
        eps = max(eps_end, eps_decay*eps) # decay epsilon : explore a bit less

        if episode % 100 == 0:
            print(f"AVG score episode {episode}: {np.mean(score_episodes[-100:])}")
                


