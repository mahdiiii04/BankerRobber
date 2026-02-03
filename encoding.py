import numpy as np

def encode_observation(obs, max_turns):
    # hand
    hand = np.array(obs["hand"], dtype=np.int32)

    # turn
    turn = np.array([obs["turn"] / max_turns], dtype=np.float32)

    # discarded histogram
    discarded = []
    for discards in obs["discarded_cards"].values():
        hist = np.zeros(10, dtype=np.float32)
        for card in discards:
            if card > 0:
                hist[card - 1] += 1
        discarded.append(hist)

    discarded = np.concatenate(discarded)

    # one hot enc phase
    phase = obs["phase"]
    phase_vec = np.zeros(3, dtype=np.float32)
    phase_vec[phase] = 1.0

    return np.concatenate([hand, turn, discarded, phase_vec])
