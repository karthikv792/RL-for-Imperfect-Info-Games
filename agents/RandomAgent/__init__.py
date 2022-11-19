import random
class RandomPlayer():
    def __init__(self, id, team, cards_at_hand = []):
        self.name = 'Random Player ' + str(id)
        self.id = id
        self.cards_at_hand = cards_at_hand
        self.team = team
        self.no_of_sequences = []

    def make_move(self, board):
        moves = self.get_legal_moves(board)
        return random.choice(moves)