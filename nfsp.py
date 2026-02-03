import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from encoding import encode_observation
from game import BankerRobberGame
from network import NFSPNetwork
from buffers import ReplayBuffer, ReservoirBuffer


class NFSPAgent:
    def __init__(
        self,
        num_actions=5,
        card_embed_dim=16,
        hidden_dim=128,
        epsilon=0.1,
        gamma=0.99,
        br_buffer_size=10_000,
        policy_buffer_size=100_000,
        batch_size=32,
        lr=1e-3
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Network
        self.net = NFSPNetwork(
            num_actions=num_actions,
            card_emb_dim=card_embed_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # Optimizers (shared backbone, separate heads)
        self.q_optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Buffers
        self.br_buffer = ReplayBuffer(br_buffer_size)
        self.policy_buffer = ReservoirBuffer(policy_buffer_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, obs_vec, action_mask):
        obs_tensor = torch.tensor(
            obs_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        q_values, policy_logits = self.net(obs_tensor, mask_tensor)

        # ε-greedy between BR and average policy
        if np.random.rand() < self.epsilon:
            # Best response (Q-head)
            q_values = q_values.masked_fill(~mask_tensor, -1e9)
            action = torch.argmax(q_values, dim=-1).item()
            return 1, action  # BR
        else:
            # Average policy
            dist = Categorical(logits=policy_logits)
            action = dist.sample().item()
            return 0, action  # AVG

    # ------------------------------------------------------------------
    # Q-head training (DQN-style)
    # ------------------------------------------------------------------
    def train_q_head(self):
        if len(self.br_buffer.buffer) < self.batch_size:
            return

        batch = self.br_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values, _ = self.net(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q, _ = self.net(next_obs)
            max_next_q = next_q.max(dim=1)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(q_sa, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    # ------------------------------------------------------------------
    # Policy-head training (supervised learning)
    # ------------------------------------------------------------------
    def train_policy_head(self):
        if len(self.policy_buffer.buffer) < self.batch_size:
            return

        batch = self.policy_buffer.sample(self.batch_size)
        obs, actions = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)

        _, policy_logits = self.net(obs)
        loss = F.cross_entropy(policy_logits, actions)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    # ------------------------------------------------------------------
    # Main NFSP training loop
    # ------------------------------------------------------------------
    def train(self, num_episodes, max_turns=None):
        env = BankerRobberGame()

        for episode in range(num_episodes):
            env.reset()

            while env.agents:
                agent_name = env.agent_selection
                obs_dict = env.observe(agent_name)

                obs_vec = encode_observation(
                    obs_dict["observation"],
                    max_turns=env.max_turns if max_turns is None else max_turns
                )

                action_mask = obs_dict["action_mask"]

                mode, action = self.select_action(obs_vec, action_mask)

                env.step(action)

                reward = env.rewards[agent_name]
                done = env.terminations[agent_name] or env.truncations[agent_name]

                if not done:
                    next_obs_dict = env.observe(agent_name)
                    next_obs_vec = encode_observation(
                        next_obs_dict["observation"],
                        max_turns=env.max_turns if max_turns is None else max_turns
                    )
                else:
                    next_obs_vec = np.zeros_like(obs_vec)

                # Store transition
                if mode == 1:
                    self.br_buffer.push(
                        obs_vec, action, reward, next_obs_vec, float(done)
                    )
                else:
                    self.policy_buffer.push(obs_vec, action)

                # Train
                self.train_q_head()
                self.train_policy_head()

            if (episode + 1) % 50 == 0:
                print(
                    f"[NFSP] Episode {episode + 1}/{num_episodes} | "
                    f"BR buffer: {len(self.br_buffer.buffer)} | "
                    f"Policy buffer: {len(self.policy_buffer.buffer)}"
                )
