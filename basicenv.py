# Create an openAI gym like environment
import numpy as np
import random
import time
import sys
import os
import math
from copy import deepcopy
import pygame


class Board():
    def __init__(self):
        self.board_size = 800
        self.pic_size = (int(self.board_size * 0.0654), self.board_size // 10)
        self.coin_size = (int(self.board_size * 0.0327), self.board_size // 20)
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

    def visualize(self):
        self.screen = pygame.display.set_mode([self.board_size, self.board_size])

    def get_legal_moves(self, player):
        legal_moves = []
        for ind_row, row in enumerate(self.board_positions):
            for ind_col, col in enumerate(row):
                if self.board_positions[ind_row][ind_col] == '':
                    continue
                if self.coin_positions[ind_row][ind_col] == 0:
                    if col in player.cards_at_hand:
                        legal_moves.append(((ind_row, ind_col),
                                            '0pos'))  # Will yield all positions where the player has cards to put
                        # the token
                    else:
                        if len([x for x in player.cards_at_hand if 'J2' in x]):
                            legal_moves.append(((ind_row, ind_col),
                                                '0posJ2'))  # Will yield all positions where the player needs J2 to
                            # put the token
                else:
                    if self.board_positions[ind_row][ind_col] != '' and self.coin_positions[ind_row][ind_col] != \
                            player.team[1]:
                        if len([x for x in player.cards_at_hand if 'J1' in x]):
                            legal_moves.append(((ind_row, ind_col), '1pos'))
        return legal_moves

    def change_corners_for_win_check(self, player):
        self.coin_positions[0][0], self.coin_positions[0][9], self.coin_positions[9][0], \
        self.coin_positions[9][9] = [player.team[1]] * 4

    def is_win(self, player):
        self.change_corners_for_win_check(player)
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
            player.cards_at_hand.remove(self.board_positions[move[0][0]][move[0][1]])
            # key = (self.board_positions[move[0][0]][move[0][1]], move[0][0],move[0][1])
            # x_value, y_value = self.board_pics[key][1]
            print("Putting on ", self.board_positions[move[0][0]][move[0][1]])
            # picture = self.get_coin_picture(self.team)
            # self.screen.blit(picture, (x_value + self.coin_size[0] / 2, y_value + self.coin_size[1] / 2))
            # pygame.display.update()
        if move[1] == '1pos':
            self.coin_positions[move[0][0]][move[0][1]] = 0
            player.cards_at_hand.remove([x for x in player.cards_at_hand if 'J1' in x][0])
            # key = (self.board_positions[c_row][c_col], c_row, c_col)
            # (pic, (x_value, y_value)) = self.board_pics[key]
            print("Removing ", self.board_positions[move[0][0]][move[0][1]])
            # self.screen.blit(pic, (x_value, y_value))
            # pygame.display.update()
        if move[1] == '0posJ2':
            self.coin_positions[move[0][0]][move[0][1]] = player.team[1]
            player.cards_at_hand.remove([x for x in player.cards_at_hand if 'J2' in x][0])
            # key = (self.board_positions[move[0][0]][move[0][1]], move[0][0],move[0][1])
            # x_value, y_value = self.board_pics[key][1]
            print("Putting (with J2) on ", self.board_positions[move[0][0]][move[0][1]])
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
    def __init__(self, id, team):
        self.id = id
        self.cards_at_hand = []
        self.team = team
        self.no_of_sequences = []

    def make_move(self, board):
        moves = board.get_legal_moves(self)
        return random.choice(moves)


"""
Initiliazing the environment with two players
player 1 will play

"""


class Env():
    def __init__(self, players):
        self.board = Board()
        self.players = players
        self.deck = Deck()
        self.discarded_cards = []
        self.no_of_cards = 7
        self.action_space = []
        self.observation_space = []
        self.split_cards()

    def step(self, player, action, game_over=False):
        """

        :param player:
        :param action:
        :return: observation, reward, done, info
        """
        if game_over:
            return self.board.coin_positions, -1, True, {}
        self.board.execute_move(player, action)
        self.discarded_cards.append(self.board.board_positions[action[0][0]][action[0][1]])
        try:
            player.cards_at_hand.append(self.deck.card_deck.pop(0))
        except IndexError:
            print("Reshuffling discarded cards in deck")
            print(self.discarded_cards)
            self.deck.card_deck = self.discarded_cards.copy()
            random.shuffle(self.deck.card_deck)
            player.cards_at_hand.append(self.deck.card_deck.pop(0))
        done = self.board.is_win(player)
        if done:
            return self.board.coin_positions, 1, True, {}
        elif len(self.board.get_legal_moves(player)):
            return self.board.coin_positions, 0, False, {}
        else:
            return self.board.coin_positions, 1e-4, True, {}

    def reset(self):
        self.board = Board()
        self.players = players
        self.deck = Deck()
        self.discarded_cards = []
        self.no_of_cards = 7
        self.action_space = []
        self.observation_space = []
        self.split_cards()

    def split_cards(self):
        for i in range(self.no_of_cards):
            for j in self.players:
                j.cards_at_hand.append(self.deck.card_deck.pop(0))

    def get_coin_picture(self, team):
        if team[1] == 1:
            return pygame.transform.scale(pygame.image.load('coins/red.png').convert(), team[0])
        else:
            return pygame.transform.scale(pygame.image.load('coins/blue.png').convert(), team[0])

    def display_board(self):
        self.board.visualize()
        for i in range(10):
            for j in range(10):
                if (self.board.board_positions[i][j] != ''):
                    pic = pygame.image.load('cards/' + self.board.board_positions[i][j] + '.png').convert()
                    pic = pygame.transform.scale(pic, self.board.pic_size)
                    # print(pic.get_rect())
                    x_value = j * self.board.board_size / 10
                    y_value = i * self.board.board_size / 10
                    self.board.board_pics[(self.board.board_positions[i][j], i, j)] = [pic, (x_value, y_value)]
                    self.board.screen.blit(pic, (x_value, y_value))
                    # self.board.screen.blit(coin_pic,(x_value+self.board.coin_size[0]/2,y_value+self.board.coin_size[1]/2))
                    # self.board.screen.blit(coin_pic,(x_value,y_value))
                    if self.board.coin_positions[i][j] != 0:
                        key = (self.board.board_positions[i][j], i, j)
                        x_value, y_value = self.board.board_pics[key][1]
                        # print("Putting (with J2) on ", self.board_positions[i][j])
                        if self.board.coin_positions[i][j] == 1:
                            team = self.board.red_coin
                        else:
                            team = self.board.blue_coin
                        self.board.screen.blit(self.get_coin_picture(team),
                                          (x_value + self.board.coin_size[0] / 2, y_value + self.board.coin_size[1] / 2))
                        pygame.display.update()
        pygame.image.save(self.board.screen, "winning.jpg")
