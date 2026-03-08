# agents/ismcts/ismcts.py
from __future__ import annotations
import random
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index
from engine.board import BOARD_LAYOUT, Occupant
from engine.deck import SEQUENCE_DECK
from agents.ismcts.mcts import mcts_search, PolicyValueFn


def determinize(
    info_set: InformationSet,
    player: int,
    seed: int | None = None,
) -> GameState:
    """Create a concrete GameState from an InformationSet by sampling opponent's hand."""
    rng = random.Random(seed)
    opponent = 2 if player == 1 else 1

    # All cards in the deck (as strings)
    all_cards = [str(c) for c in SEQUENCE_DECK]

    # Remove cards placed on the board
    for r in range(10):
        for c in range(10):
            if info_set.occupancy[r][c] in (Occupant.PLAYER1, Occupant.PLAYER2):
                card_str = BOARD_LAYOUT[r][c]
                if card_str and card_str in all_cards:
                    all_cards.remove(card_str)

    # Remove own hand from available pool
    available = list(all_cards)
    for card in info_set.own_hand:
        if card in available:
            available.remove(card)
    # Remove discard pile from available pool
    for card in info_set.discard_pile:
        if card in available:
            available.remove(card)

    rng.shuffle(available)

    # Deal opponent hand
    hand_size = len(info_set.own_hand)
    opponent_hand = tuple(available[:hand_size])
    remaining_deck = tuple(available[hand_size:])

    return GameState(
        occupancy=info_set.occupancy.copy(),
        current_player=info_set.current_player,
        hands={
            player: info_set.own_hand,
            opponent: opponent_hand,
        },
        deck_cards=remaining_deck,
        discard_pile=info_set.discard_pile,
        sequences=info_set.sequences,
    )


def ismcts_search(
    info_set: InformationSet,
    player: int,
    policy_value_fn: PolicyValueFn,
    num_determinizations: int = 20,
    num_simulations: int = 100,
    cpuct: float = 1.5,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
) -> Action:
    """Run ISMCTS: determinize, run MCTS on each, aggregate visit counts."""
    aggregated: dict[int, int] = {}  # action_index -> total visits

    for d in range(num_determinizations):
        det_state = determinize(info_set, player, seed=d)
        _, visit_counts = mcts_search(
            state=det_state,
            policy_value_fn=policy_value_fn,
            num_simulations=num_simulations,
            cpuct=cpuct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )
        for action, visits in visit_counts.items():
            idx = action_to_index(action)
            aggregated[idx] = aggregated.get(idx, 0) + visits

    best_idx = max(aggregated, key=aggregated.get)
    from engine.actions import index_to_action
    return index_to_action(best_idx)
