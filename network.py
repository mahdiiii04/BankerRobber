import torch
import torch.nn as nn
import torch.nn.functional as F

class NFSPNetwork(nn.Module):
    def __init__(self, num_actions, card_emb_dim=16, hidden_dim=128):
        super().__init__()

        # Hand Embedding
        self.card_embedder = nn.Embedding(
            num_embeddings=11,
            embedding_dim=card_emb_dim
        )

        # Shared Encoder
        self.fc1 = nn.Linear(44 + 5 * card_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Best response (RL)
        self.q_head = nn.Linear(hidden_dim, num_actions)

        # Average Policy head (supervised)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs_vec, action_mask=None):

        hand = obs_vec[:, :5].long()
        rest = obs_vec[:, 5:]

        hand_emb = self.card_embedder(hand)
        hand_emb = hand_emb.view(hand_emb.size(0), -1)

        x = torch.cat([hand_emb, rest], dim=-1)

        # shared layyers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # heads
        q_values = self.q_head(x)  # best-response
        policy_logits = self.policy_head(x) # average policy

        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)
        
        return q_values, policy_logits
