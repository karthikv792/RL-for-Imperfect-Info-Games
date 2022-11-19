from agents.BasePlayer import Player
import math
import random

class BasePolicyAgent(Player):
    def __init__(self, id, team):
        self.name = 'Base Policy Player ' + str(id)
        self.id = id
        self.cards_at_hand = []
        self.team = team
        self.no_of_sequences = []

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_player_pos(self, board):
        board_pos_player = []
        for i in range(10):
            for j in range(10):
                if (i, j) in [(0, 0), (0, 9), (9, 0), (9, 9)]:
                    continue
                if board.coin_positions[i][j] == self.team[1]:
                    board_pos_player.append((i, j))
        return board_pos_player

    def make_move(self, board, discarded_cards=[]):
        moves = self.get_legal_moves(board)
        moves_put = [i for i in moves if i[1] != '1pos']
        moves_remove = [i for i in moves if i[1] == '1pos']
        # Get all the board positions of this player
        # Get all the board positions of the opponent
        board_pos_player = self.get_player_pos(board)
        if board_pos_player == []:
            return random.choice(moves)
        global_min = math.inf
        global_min_move = None
        for move in moves_put:
            pos1 = move[0]
            local_min = math.inf
            for pos2 in board_pos_player:
                local_min = min(local_min, self.manhattan_distance(pos1, pos2))
            if local_min < global_min:
                global_min = local_min
                global_min_move = move
        global_remove_min = math.inf
        global_remove_min_move = None
        for move in moves_remove:
            pos1 = move[0]
            local_min = math.inf
            for pos2 in board_pos_player:
                local_min = min(local_min, self.manhattan_distance(pos1, pos2))
            if local_min < global_remove_min:
                global_remove_min = local_min
                global_remove_min_move = move
        if global_min < global_remove_min:
            return global_min_move
        else:
            return global_remove_min_move