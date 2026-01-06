import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from collections import Counter
import functools

PHASE_MAPPING = {
    "discard": 0,
    "vote": 1,
    "player_voting": 2
}

class BankerRobberGame(AECEnv):
    metadata = {"render_modes": ["human"], "name": "robber_banker_game"}

    def __init__(self, render_mode=None):
        super().__init__()
        self._num_agents = 4
        self.agents = [f"player_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]
        self.render_mode = render_mode
        self.max_turns = 10
        self.current_turn = 0

        self.robber_index = None

        self.deck = []
        self.hands = {agent: [] for agent in self.agents}

        self.votes = {}
        self.phase = None
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({

            "observation": spaces.Dict({
                "hand": spaces.MultiDiscrete([11] * 5),
                "turn": spaces.Discrete(self.max_turns + 1),
                "discarded_cards": spaces.Dict({
                    agent: spaces.MultiDiscrete([11] * 10) for agent in self.agents
                }),
                "phase": spaces.Discrete(3) # 0 for discarding phase and 1 for voting phase, 2 for player voting phase
            }),
            "action_mask": spaces.Box(low=0, high=1, shape=(max(5, self._num_agents),), dtype=np.int8)
            })
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(max(5, self._num_agents))
    
    def render(self):
        if self.render_mode != "human":
            return

        print("\n" + "=" * 40)
        print(f"Turn: {self.current_turn + 1}/{self.max_turns}")
        print(f"Phase: {self.phase.upper()}")
        print(f"Robber: player_{self.robber_index} (hidden in real play)")
        print("-" * 40)

        # Show each player's hand and discard pile
        for agent in self.possible_agents:
            hand_str = " ".join(str(c) for c in self.hands.get(agent, []))
            discard_str = " ".join(str(c) for c in self.discarded_piles.get(agent, []))
            print(f"{agent} | Hand: {hand_str} | Discards: {discard_str}")

        # Show votes depending on phase
        if self.phase == "vote":
            print("\nVotes so far:")
            for agent in self.agents:
                if self.votes[agent]:
                    print(f"{agent}: {'Stop' if self.votes[agent][-1] == 1 else 'Continue'}")
                else:
                    print(f"{agent}: No vote yet")

        elif self.phase == "player_voting":
            print("\nPlayer voting so far:")
            for agent in self.agents:
                if self.player_votes[agent]:
                    voted_for = self.player_votes[agent][-1]
                    print(f"{agent} voted for player_{voted_for}")
                else:
                    print(f"{agent}: No vote yet")

        print("=" * 40 + "\n")


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_turn = 0
        self.phase = "discard"
        self.agents = self.possible_agents[:]
        self.penalties = {agent: 0 for agent in self.agents}
        self.hands = {agent: [] for agent in self.agents}

        self.deck = list(range(1, 11)) * (self._num_agents + 2)
        np.random.shuffle(self.deck)

        self.robber_index = np.random.randint(0, self._num_agents)

        for agent in self.agents:
            if agent == self.agents[self.robber_index]:
                self.hands[agent] = [0] + [self.deck.pop() for _ in range(4)]
            else:
                self.hands[agent] = [self.deck.pop() for _ in range(5)]
        
        self.discarded_piles = {agent: [0] * 10 for agent in self.agents}
        self.votes = {agent: [] for agent in self.agents}
        self.player_votes = {agent: [] for agent in self.agents}

        self.agent_selection = self.agents[0]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
    
    def observe(self, agent):
        return {
            "observation": {
                "hand": self.hands[agent],
                "turn": self.current_turn,
                "discarded_cards": {agent: self.discarded_piles[agent] for agent in self.agents},
                "phase": PHASE_MAPPING[self.phase]
            },
            "action_mask": self.get_action_mask(agent)
        }
    
    def get_action_mask(self, agent):
        size = max(5, self._num_agents)

        if self.phase == "discard":
            mask = np.zeros(size, dtype=np.int8)
            mask[:5] = 1
            if agent == self.agents[self.robber_index]:
                robber_index = self.hands[agent].index(0)
                mask[robber_index] = 0
        elif self.phase == "vote":
            mask = np.zeros(size, dtype=np.int8)
            mask[:2] = 1
        elif self.phase == "player_voting":
            mask = np.zeros(size, dtype=np.int8)
            mask[:self._num_agents] = 1

        return mask

    def step(self, action):
        agent = self.agent_selection

        if self.phase == "discard":
            discarded_card = self.hands[agent][action]
            if discarded_card == 0:
                print(f"Player {agent} cannot discard the robber card.")
                self.penalties[agent] += -10
                self._cumulative_rewards[agent] += -10
                self._advance_turn()
                return
            self.discarded_piles[agent][self.current_turn] = discarded_card
            self.hands[agent][action] = self.deck.pop()
        elif self.phase == "vote":
            if action > 1:
                print(f"Invalid vote action by {agent}.")
                self.penalties[agent] += -1
                self._cumulative_rewards[agent] += -1
                self.votes[agent].append(0)
                self._advance_turn()
                return
            self.votes[agent].append(action)
        elif self.phase == "player_voting":
            if action > self._num_agents - 1:
                print(f"Invalid player voting action by {agent}.")
                self.penalties[agent] += -1
                self._cumulative_rewards[agent] += -1
                agent_index = self.agents.index(agent)
                self.player_votes[agent].append(agent_index)
                self._advance_turn()
                return
            self.player_votes[agent].append(action)
        self._advance_turn()
    
    def _advance_turn(self):
        idx = self.agents.index(self.agent_selection)
        if idx + 1 < len(self.agents):
            self.agent_selection = self.agents[idx + 1]
        else:
            if self.phase == "discard":
                if self.current_turn == self.max_turns - 1:
                    self.phase = "player_voting"
                    self.agent_selection = self.agents[0]
                else:
                    self.phase = "vote"
                    self.agent_selection = self.agents[0]
            elif self.phase == "vote":
                votes = [self.votes[agent][-1] for agent in self.agents]
                if votes.count(1) == self._num_agents:
                    self.phase = "player_voting"
                    self.agent_selection = self.agents[0]
                else:
                    self.current_turn += 1
                    self.phase = "discard"
                    self.agent_selection = self.agents[0]
            elif self.phase == "player_voting":
                votes = [self.player_votes[agent][-1] for agent in self.agents]
                counts = Counter(votes)
                voted_1 = counts.most_common(1)[0][1]
                voted_2 = counts.most_common(2)[1][1] if len(counts) > 1 else 0
                if voted_1 > voted_2:
                    if counts.most_common(1)[0][0] == self.robber_index:
                        self._calculate_winner(case="bankers_win")
                    else:
                        self._calculate_winner(case="robber_wins")
                else:
                    if self.current_turn == self.max_turns - 1:
                        self._calculate_winner(case="robber_wins")
                    else:
                        self.phase = "discard"
                        self.current_turn += 1
                        self.agent_selection = self.agents[0]
    
    def _calculate_winner(self, case):
        if case == "bankers_win":
            total_sum = sum([sum(self.discarded_piles[agent]) for agent in self.agents])
            total_sum = total_sum / (self.current_turn  + 1)
            for agent in self.agents:
                if agent == self.agents[self.robber_index]:
                    self._cumulative_rewards[agent] += -total_sum
                else:
                    self._cumulative_rewards[agent] += total_sum / (self._num_agents - 1)
        elif case == "robber_wins":
            robber_sum = sum(self.hands[self.agents[self.robber_index]])
            for agent in self.agents:
                if agent == self.agents[self.robber_index]:
                    self._cumulative_rewards[agent] += robber_sum
                else:
                    self._cumulative_rewards[agent] += -robber_sum
        self.rewards = self._cumulative_rewards.copy()
        self.end_game()

    def end_game(self):
        self.terminations = {agent: True for agent in self.agents}
        self.truncations = {agent: True for agent in self.agents}
        self.agents = []

            