# BankerRobberGame README

## Project Overview

**BankerRobberGame** is a custom multi-agent sequential game environment implemented as a PettingZoo `AECEnv` for multi-agent reinforcement learning (MARL) research. The game is designed to study **deception**, **theory-of-mind**, and **higher-order recursive reasoning** in strategic interactions.

The environment models a variant of the classic "Banker-Robber" setup with 4 players:
- 3 honest **Bankers** try to accumulate value by discarding high cards over multiple turns.
- 1 hidden **Robber** (randomly assigned) holds a special "0" card and aims to maximize their final hand value without being identified.

The game proceeds in three phases:
1. **Discard phase** – Players discard one card and draw a new one.
2. **Vote phase** – Players vote to "Continue" or "Stop" the game.
3. **Player voting phase** – If voting ties or reaches the final turn, players vote to accuse a suspected Robber.

The outcome depends on whether the majority correctly identifies the Robber and the accumulated/discarded values.

### Research Motivation

This environment serves as a **controlled benchmark** for measuring:
- Levels of theory-of-mind (belief tracking about others' beliefs).
- Deception strategies (Robber hiding identity).
- Recursive reasoning (e.g., "I think that you think that I am the Robber").
- Emergence of bluffing, signaling, and coordinated accusation in imperfect-information settings.

It is particularly suited for studying **human-like strategic behaviors** in MARL and game-theoretic analysis of non-zero-sum sequential games with hidden roles.

## Installation

```bash
pip install pettingzoo gymnasium numpy
```

No additional dependencies are required.

## Usage Example

```python
from banker_robber_game import BankerRobberGame

env = BankerRobberGame(render_mode="human")  # or None
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # Your agent policy here
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)  # random valid action for demo
    
    env.step(action)
    
env.close()
```

## Observation Space

Each agent receives a dictionary observation:
- `hand`: List of 5 cards (MultiDiscrete)
- `turn`: Current turn (0 to max_turns)
- `discarded_cards`: Dictionary of all players' discard histories
- `phase`: Current phase (0=discard, 1=vote, 2=player_voting)
- `action_mask`: Valid actions for current phase

## Action Space

Discrete actions depending on phase:
- Discard phase: Choose index in hand to discard (masked to prevent discarding Robber card)
- Vote phase: 0=Continue, 1=Stop
- Player voting phase: Choose player index to accuse

## Rendering

Set `render_mode="human"` for text-based visualization of hands, discards, votes, and current phase.

---

This environment was developed as part of ongoing research into human-inspired strategic reasoning in multi-agent reinforcement learning.
