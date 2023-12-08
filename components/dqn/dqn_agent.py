import numpy as np
import random
import torch
import torch.optim as optim
from collections import deque


from components.dqn.dqn_model import DQN
from components.tools.utils import GAMMA, BATCH_SIZE, TAU, UPDATE_EVERY, LEARNING_RATE, MEMORY_SIZE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.learning_rate = LEARNING_RATE

        self.local_network = DQN(state_size, action_size).to(DEVICE)
        self.target_network = DQN(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=self.learning_rate)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = random.sample(self.memory, BATCH_SIZE)
            self.learn(experiences)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)

        local_max_actions = self.local_network(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.target_network(next_states).detach().gather(1, local_max_actions)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        Q_expected = self.local_network(states).gather(1, actions)

        loss = torch.nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_network, self.target_network)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
