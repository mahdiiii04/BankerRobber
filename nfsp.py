# nfsp.py (with additional fixes: reward normalization, Huber loss, grad clip, lower lr, higher beta, longer target update)
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.utils as utils

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
        lr=1e-5,
        beta=0.1,  # Increased for more entropy
        target_update_every=1000,  # Increased
        reward_scale=1.0 / 50.0,  # Normalize rewards by approx max |r| ~50
        grad_clip=1.0
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.beta = beta
        self.target_update_every = target_update_every
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip
        self.steps = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Network
        self.net = NFSPNetwork(
            num_actions=num_actions,
            card_emb_dim=card_embed_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.target_net = NFSPNetwork(
            num_actions=num_actions,
            card_emb_dim=card_embed_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        # Optimizers (shared backbone, separate heads)
        self.q_optimizer = optim.Adam(
            list(self.net.fc1.parameters()) +
            list(self.net.fc2.parameters()) +
            list(self.net.q_head.parameters()),
            lr=lr
        )
        
        self.policy_optimizer = optim.Adam(
            list(self.net.fc1.parameters()) +
            list(self.net.fc2.parameters()) +
            list(self.net.policy_head.parameters()),
            lr=lr
        )

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
        obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        masks = torch.tensor(masks, dtype=torch.bool, device=self.device)
        next_masks = torch.tensor(next_masks, dtype=torch.bool, device=self.device)

        q_values, _ = self.net(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q, _ = self.target_net(next_obs)
            next_q = next_q.masked_fill(~next_masks, -1e9)
            max_next_q = next_q.max(dim=1)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_sa, target)  # Huber loss

        self.q_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            utils.clip_grad_norm_(
                list(self.net.fc1.parameters()) +
                list(self.net.fc2.parameters()) +
                list(self.net.q_head.parameters()),
                self.grad_clip
            )
        self.q_optimizer.step()

    # ------------------------------------------------------------------
    # Policy-head training (supervised learning)
    # ------------------------------------------------------------------
    def train_policy_head(self):
        if len(self.policy_buffer.buffer) < self.batch_size:
            return

        batch = self.policy_buffer.sample(self.batch_size)
        obs, actions, masks = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.bool, device=self.device)

        _, policy_logits = self.net(obs, masks)
        loss_ce = F.cross_entropy(policy_logits, actions)
        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        loss = loss_ce - self.beta * entropy

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            utils.clip_grad_norm_(
                list(self.net.fc1.parameters()) +
                list(self.net.fc2.parameters()) +
                list(self.net.policy_head.parameters()),
                self.grad_clip
            )
        self.policy_optimizer.step()

    # ------------------------------------------------------------------
    # Main NFSP training loop
    # ------------------------------------------------------------------
    def train(self, num_episodes, max_turns=None):
        env = BankerRobberGame()

        episode_rewards = []  # To log average rewards

        for episode in range(num_episodes):
            env.reset()
            probe_obs_vec = None
            last_obs_vec = {}
            last_action = {}
            last_mode = {}
            total_reward_seen = {agent: 0 for agent in env.possible_agents}
            episode_reward = 0

            while env.agents:
                agent_name = env.agent_selection
                obs_dict = env.observe(agent_name)

                obs_vec = encode_observation(
                    obs_dict["observation"],
                    max_turns=env.max_turns if max_turns is None else max_turns
                )

                if probe_obs_vec is None:
                    probe_obs_vec = obs_vec.copy()

                action_mask = obs_dict["action_mask"]
                mode, action = self.select_action(obs_vec, action_mask)

                env.step(action)

                reward = env.rewards[agent_name]
                done = env.terminations[agent_name] or env.truncations[agent_name]
                total_reward_seen[agent_name] += reward
                episode_reward += reward  # Sum over all for logging

                last_obs_vec[agent_name] = obs_vec
                last_action[agent_name] = action
                last_mode[agent_name] = mode

                if not done and agent_name in env.agents:
                    next_obs_dict = env.observe(agent_name)
                    next_obs_vec = encode_observation(
                        next_obs_dict["observation"],
                        max_turns=env.max_turns if max_turns is None else max_turns
                    )
                    next_action_mask = next_obs_dict["action_mask"]
                else:
                    next_obs_vec = np.zeros_like(obs_vec)
                    next_action_mask = np.zeros_like(action_mask)

                # Normalize reward
                norm_reward = reward * self.reward_scale

                # Store transition
                if mode == 1:  # Best-response
                    self.br_buffer.add(
                        (obs_vec, action, norm_reward, next_obs_vec, float(done), action_mask, next_action_mask)
                    )
                else:          # Average policy
                    self.policy_buffer.add((obs_vec, action, action_mask))

                # Train both heads
                self.train_q_head()
                self.steps += 1
                if self.steps % self.target_update_every == 0:
                    self.target_net.load_state_dict(self.net.state_dict())
                self.train_policy_head()

            # Add terminal transitions for additional rewards if any
            for agent in env.possible_agents:
                additional_r = env._cumulative_rewards[agent] - total_reward_seen[agent]
                if additional_r != 0:
                    norm_additional_r = additional_r * self.reward_scale
                    if last_mode.get(agent, 0) == 1:
                        next_obs_vec = np.zeros_like(last_obs_vec[agent])
                        next_action_mask = np.zeros_like(action_mask)  # dummy
                        self.br_buffer.add(
                            (last_obs_vec[agent], last_action[agent], norm_additional_r, next_obs_vec, 1.0, action_mask, next_action_mask)
                        )
                    episode_reward += additional_r  # For logging

            episode_rewards.append(episode_reward / len(env.possible_agents))  # Avg per agent

            # ===== Logging & diagnostics =====
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                print(
                    f"[NFSP] Episode {episode + 1}/{num_episodes} | "
                    f"BR buffer: {len(self.br_buffer.buffer)} | "
                    f"Policy buffer: {len(self.policy_buffer.buffer)} | "
                    f"Avg reward: {avg_reward:.2f}"
                )

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(
                        probe_obs_vec.astype(np.float32)
                    ).unsqueeze(0).to(self.device)

                    q, pi_logits = self.net(obs_tensor)
                    pi = torch.softmax(pi_logits, dim=-1)

                    print("   Q-values:", q.cpu().numpy())
                    print("   π (avg policy):", pi.cpu().numpy())