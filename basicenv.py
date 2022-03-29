# Create an openAI gym like environment
import numpy as np
import random
import time
import sys
import os
import math


class Board():
    def __init__(self):
        self.board_size = 800
        self.pic_size = (int(self.board_size * 0.0654), self.board_size // 10)
        self.coin_size = (int(self.board_size * 0.0327), self.board_size // 20)
        self.screen = pygame.display.set_mode([self.board_size, self.board_size])
        self.board_positions = [['', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', ''],
                                ['6C', '5C', '4C', '3C', '2C', 'AH', 'KH', 'QH', '10H', '10S'],
                                ['7C', 'AS', '2D', '3D', '4D', '5D', '6D', '7D', '9H', 'QS'],
                                ['8C', 'KS', '6C', '5C', '4C', '3C', '2C', '8D', '8H', 'KS'],
                                ['9C', 'QS', '7C', '6H', '5H', '4H', 'AH', '9D', '7H', 'AS'],
                                ['10C', '10S', '8C', '7H', '2H', '3H', 'KH', '10D', '6H', '2D'],
                                ['QC', '9S', '9C', '8H', '9H', '10H', 'QH', 'QD', '5H', '3D'],
                                ['KC', '8S', '10C', 'QC', 'KC', 'AC', 'AD', 'KD', '4H', '4D'],
                                ['AC', '7S', '6S', '5S', '4S', '3S', '2S', '2H', '3H', '5D'],
                                ['', 'AD', 'KD', 'QD', '10D', '9D', '8D', '7D', '6D', '']]
        self.board_pics = {}
        self.red_coin = (self.coin_size, 1)
        self.blue_coin = (self.coin_size, 2)
        self.coin_positions = [[0] * 10 for _ in range(10)]
        # self.coin_positions[0][0],self.coin_positions[9][0],self.coin_positions[0][9],self.coin_positions[9][9],=3,3,3,3
        # self.initialize_board()

    def get_legal_moves(self, player):
        legal_moves = []
        for ind_row, row in enumerate(self.board_positions):
            for ind_col, col in enumerate(row):
                if board.coin_positions[ind_row][ind_col] == 0:
                    if col in player.cards_at_hand:
                        legal_moves.append(((ind_row, ind_col),
                               '0pos'))  # Will yield all positions where the player has cards to put the token
                    else:
                        if len([x for x in player.cards_at_hand if 'J2' in x]):
                            legal_moves.append(((ind_row, ind_col),
                                   '0posJ2'))  # Will yield all positions where the player needs J2 to put the token
                else:
                    if board.board_positions[ind_row][ind_col] != '' and board.coin_positions[ind_row][ind_col] != \
                            player.team[1]:
                        if len([x for x in self.cards_at_hand if 'J1' in x]):
                            legal_moves.append(((ind_row, ind_col), '1pos'))
        return legal_moves

    def is_win(self, player):
        check_token1 = player.team[1]
        win_count = 5
        transpose_coin_positions = np.array(deepcopy(self.coin_positions)).T.tolist()
        for i in range(10):
            for j in range(win_count):
                # If board has 5 in a row
                if all([token1 == check_token1 for token1 in self.coin_positions[i][j:j + win_count]]):
                    print("ROW")
                    if (((i, j), (i, j + win_count - 1)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i, j), (i, j + win_count - 1)))
                    if len(player.no_of_sequences) == 2:
                        return True
                # If the board has 5 in a col
                if all([token1 == check_token1 for token1 in transpose_coin_positions[i][j:j + win_count]]):
                    print("Column")
                    if (((i, j), (i + win_count - 1, j)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i, j), (i + win_count - 1, j)))
                    if len(player.no_of_sequences) == 2:
                        return True
                # If the board has 5 diagonally
        i = 0
        j = 0
        while i < 10:
            # next element
            count_diag = 0
            row, col = i, j
            while row < 10 and col > -1:
                if self.coin_positions[row][col] == check_token1:
                    count_diag += 1
                else:
                    count_diag = 0
                if count_diag == win_count:
                    print("Diag,left")
                    if (((i, j), (row, col)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i, j), (row, col)))
                    if len(player.no_of_sequences) == 2:
                        return True
                row += 1
                col -= 1
            count_diag = 0
            row, col = i, j
            while row < 10 and col < 10:
                if self.coin_positions[row][col] == check_token1:
                    count_diag += 1
                else:
                    count_diag = 0
                row += 1
                col += 1
                if count_diag == win_count:
                    print("Diag,right")
                    if (((i, j), (row, col)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i, j), (row, col)))
                    if len(player.no_of_sequences) == 2:
                        return True
            if j < 9:
                j += 1
            else:
                i += 1
        return False

    def execute_move(self, player, move):
        if move[1] == '0pos':
            self.coin_positions[move[0][0]][move[0][1]] = player.team[1]
            player.cards_at_hand.remove(move[0][0])
            # key = (self.board_positions[c_row][c_col], c_row, c_col)
            # x_value, y_value = self.board_pics[key][1]
            print("Putting on ", self.board_positions[c_row][c_col])
            # picture = self.get_coin_picture(self.team)
            # self.screen.blit(picture, (x_value + self.coin_size[0] / 2, y_value + self.coin_size[1] / 2))
            # pygame.display.update()
        if move[1] == '1pos':
            self.coin_positions[c_row][c_col] = 0
            player.cards_at_hand.remove([x for x in player.cards_at_hand if 'J1' in x][0])
            # key = (self.board_positions[c_row][c_col], c_row, c_col)
            # (pic, (x_value, y_value)) = self.board_pics[key]
            print("Removing ", self.board_positions[c_row][c_col])
            # self.screen.blit(pic, (x_value, y_value))
            # pygame.display.update()
        if move[1] == '0posJ2':
            self.coin_positions[c_row][c_col] = player.team[1]
            player.cards_at_hand.remove([x for x in player.cards_at_hand if 'J2' in x][0])
            # key = (self.board_positions[c_row][c_col], c_row, c_col)
            # x_value, y_value = self.board_pics[key][1]
            print("Putting (with J2) on ", self.board_positions[c_row][c_col])
            # self.screen.blit(self.get_coin_picture(player.team),
            #                   (x_value + self.coin_size[0] / 2, y_value + self.coin_size[1] / 2))
            # pygame.display.update()

class Deck():
    def __init__(self):
        self.card_deck = []
        self.add_cards()

    def add_cards(self):
        for i in range(2):
            for suit in ['C', 'S', 'H', 'D']:
                for value in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J1', 'J2', 'Q', 'K']:
                    if value == 'J2' and (suit == 'S' or suit == 'H'):
                        continue
                    if value == 'J1' and (suit == 'C' or suit == 'D'):
                        continue
                    self.card_deck.append(value + suit)
        random.shuffle(self.card_deck)


class Player():
    def __init__(self, id, team, type):
        self.id = id
        self.cards_at_hand = []
        self.team = team
        self.type = type
        self.no_of_sequences = []

    # Make move function
    def find_positions(self, board):
        for ind_row, row in enumerate(board.board_positions):
            for ind_col, col in enumerate(row):
                if board.coin_positions[ind_row][ind_col] == 0:
                    if col in self.cards_at_hand:
                        yield ((ind_row, ind_col),
                               '0pos')  # Will yield all positions where the player has cards to put the token
                    else:
                        yield ((ind_row, ind_col),
                               '0posJ2')  # Will yield all positions where the player needs J2 to put the token
                else:
                    if board.board_positions[ind_row][ind_col] != '' and board.coin_positions[ind_row][ind_col] != \
                            self.team[1]:
                        yield ((ind_row, ind_col), '1pos')

    def get_coin_picture(self, team):
        if team[1] == 1:
            return pygame.transform.scale(pygame.image.load('coins/red.png').convert(), team[0])
        else:
            return pygame.transform.scale(pygame.image.load('coins/blue.png').convert(), team[0])

    def put_without_j2(self, board, all_free_positions):
        c_row, c_col = random.choice([position for position in all_free_positions if position[1] == '0pos'])[0]
        board.coin_positions[c_row][c_col] = self.team[1]
        self.cards_at_hand.remove(board.board_positions[c_row][c_col])
        key = (board.board_positions[c_row][c_col], c_row, c_col)
        x_value, y_value = board.board_pics[key][1]
        print("Putting on ", board.board_positions[c_row][c_col])
        picture = self.get_coin_picture(self.team)
        board.screen.blit(picture, (x_value + board.coin_size[0] / 2, y_value + board.coin_size[1] / 2))
        pygame.display.update()

    def put_with_j2(self, board, all_free_positions):
        c_row, c_col = random.choice(
            [position for position in all_free_positions if position[1] == '0pos' or position[1] == '0posJ2'])[0]
        board.coin_positions[c_row][c_col] = self.team[1]
        self.cards_at_hand.remove([x for x in self.cards_at_hand if 'J2' in x][0])
        key = (board.board_positions[c_row][c_col], c_row, c_col)
        x_value, y_value = board.board_pics[key][1]
        print("Putting (with J2) on ", board.board_positions[c_row][c_col])
        board.screen.blit(self.get_coin_picture(self.team),
                          (x_value + board.coin_size[0] / 2, y_value + board.coin_size[1] / 2))
        pygame.display.update()

    def remove_with_j1(self, board, all_free_positions):
        c_row, c_col = random.choice([position for position in all_free_positions if position[1] == '1pos'])[0]
        board.coin_positions[c_row][c_col] = 0
        self.cards_at_hand.remove([x for x in self.cards_at_hand if 'J1' in x][0])
        key = (board.board_positions[c_row][c_col], c_row, c_col)
        (pic, (x_value, y_value)) = board.board_pics[key]
        print("Removing ", board.board_positions[c_row][c_col])
        board.screen.blit(pic, (x_value, y_value))
        pygame.display.update()

    def make_move(self, board):
        all_free_positions = [position for position in self.find_positions(board)]
        valid_actions = ['DoPut']
        if len([x for x in self.cards_at_hand if 'J2' in x]):
            valid_actions.append('DoJ2')
        if len([x for x in self.cards_at_hand if 'J1' in x]) and any(i[1] == '1pos' for i in all_free_positions):
            valid_actions.append('DoJ1')
        # Type of Player
        choice_of_action = random.choice(valid_actions)
        if choice_of_action == 'DoPut':
            self.put_without_j2(board, all_free_positions)
        elif choice_of_action == 'DoJ2':
            self.put_with_j2(board, all_free_positions)
        else:
            self.remove_with_j1(board, all_free_positions)


class Env():
    def __init__(self):
        self.board = Board()
        self.players = [Player(i, board.blue_coin if i % 2 == 0 else board.red_coin, 'random') for i in
                   range(1, no_players + 1)]
        self.deck = Deck()
        self.no_of_cards = 7
        self.action_space = []
        self.observation_space = []
        self.state = 'left'
        self.reward = 0
        self.done = False
        self.action = None
        self.observation = None
        self.info = None
        self.timestep = 0

    def step(self, action):
        # One step will have the opponent's move too.
        self.action = action
        self.timestep += 1
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def split_cards(self):
        for i in range(self.no_of_cards):
            for j in self.players:
                j.cards_at_hand.append(self.deck.card_deck.pop(0))

