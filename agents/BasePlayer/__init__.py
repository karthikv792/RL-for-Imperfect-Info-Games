class Player():
    def __init__(self, id, team, cards_at_hand = []):
        self.name = 'Player ' + str(id)
        self.id = id
        self.cards_at_hand = cards_at_hand
        self.team = team
        self.no_of_sequences = []

    def get_legal_moves(self, board):
        legal_moves = []
        for ind_row, row in enumerate(board.board_positions):
            for ind_col, col in enumerate(row):
                if board.board_positions[ind_row][ind_col] == '':
                    continue
                if board.coin_positions[ind_row][ind_col] == 0:
                    if col in self.cards_at_hand:
                        legal_moves.append(((ind_row, ind_col),
                                            '0pos')) # Will yield all positions where the player has cards to put
                        # the token
                    else:
                        j2s = [x for x in self.cards_at_hand if 'J2' in x]
                        if len(j2s):
                            legal_moves.append(((ind_row, ind_col),
                                                '0posJ2'))  # Will yield all positions where the player needs J2 to
                            # put the token
                else:
                    if board.board_positions[ind_row][ind_col] != '' and board.coin_positions[ind_row][ind_col] != \
                            self.team[1]:
                        j1s = [x for x in self.cards_at_hand if 'J1' in x]
                        if len(j1s):
                            legal_moves.append(((ind_row, ind_col), '1pos'))
        return legal_moves

    def make_move(self, board):
        raise NotImplementedError

