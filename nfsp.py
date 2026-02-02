import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from encoding import encode_observation
from game import BankerRobberGame
from network import NFSPNetwork
from buffers import ReplayBuffer, ReservoirBuffer

class NFSPAgent:
    def __init__(self, num_actions=5, card_embed_dim=16, hidden_dim=128, epsilon=0.1,
                 br_buffer_size=10000, policy_buffer_size=100000, batch_size=32, lr=1e-3):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = NFSPNetwork(num_actions, card_embed_dim, hidden_dim).to(self.device)
        self.q_optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.br_buffer = ReplayBuffer(br_buffer_size)
        self.policy_buffer = ReservoirBuffer(policy_buffer_size)

        self.num_actions = num_actions

    def select_action(self, obs_vec, action_mask):
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.int8, device=self.device).unsqueeze(0)
        q_values, policy_logits = self.net(obs_tensor, mask_tensor)

        if np.random.rand() < self.epsilon:
            # Best-response action
            q_values = q_values.masked_fill(mask_tensor == 0, -1e9)
            action = torch.argmax(q_values, dim=-1).item()
            return 1, action
        else:
            # Average-policy action
            policy_dist = Categorical(logits=policy_logits)
            action = policy_dist.sample().item()
            return 0, action

    def train_q_head(self):
        if len(self.br_buffer.buffer) < self.batch_size:
            return
        batch = self.br_buffer.sample(self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        q_values, _ = self.net(obs_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values, _ = self.net(next_obs_batch)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = reward_batch + (1 - done_batch) * 0.99 * max_next_q

        loss = F.mse_loss(q_values, target_q)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def train_policy_head(self):
        if len(self.policy_buffer.buffer) < self.batch_size:
            return
        batch = self.policy_buffer.sample(self.batch_size)
        obs_batch, action_batch = zip(*batch)
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device)

        _, policy_logits = self.net(obs_batch)
        loss = F.cross_entropy(policy_logits, action_batch)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
