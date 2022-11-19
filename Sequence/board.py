import numpy as np
import random
import time
import sys
import os
import math
from copy import deepcopy
import pygame
import itertools

class Board():
    def __init__(self, logging=False):
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

        self.red_coin = (self.coin_size, 1)
        self.blue_coin = (self.coin_size, 2)
        self.coin_positions = [[0] * 10 for _ in range(10)]
        self.logging = logging
        # self.coin_positions[0][0],self.coin_positions[9][0],self.coin_positions[0][9],self.coin_positions[9][9],=3,3,3,3
        # self.initialize_board()


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
                    if self.logging:
                        print('[+] Winning condition 1: ROW')

                    if (((i, j), (i, j + win_count - 1)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i, j), (i, j + win_count - 1)))
                    if len(player.no_of_sequences) == 2:
                        return True
                # If the board has 5 in a col
                if all([token1 == check_token1 for token1 in transpose_coin_positions[i][j:j + win_count]]):
                    if self.logging:
                        print('[+] Winning condition 2: COL')
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
                    if self.logging:
                        print('[+] Winning condition 3: DIAG LEFT')
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
                    if self.logging:
                        print('[+] Winning condition 4: DIAG RIGHT')
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
        card = None
        if move[1] == '0pos':
            self.coin_positions[move[0][0]][move[0][1]] = player.team[1]
            card = self.board_positions[move[0][0]][move[0][1]]
            player.cards_at_hand.remove(card)
            # key = (self.board_positions[move[0][0]][move[0][1]], move[0][0],move[0][1])
            # x_value, y_value = self.board_pics[key][1]
            if self.logging:
                print('[+] Executed move to put', self.board_positions[move[0][0]][move[0][1]])
            # picture = self.get_coin_picture(self.team)
            # self.screen.blit(picture, (x_value + self.coin_size[0] / 2, y_value + self.coin_size[1] / 2))
            # pygame.display.update()
        if move[1] == '1pos':
            self.coin_positions[move[0][0]][move[0][1]] = 0
            card  = [x for x in player.cards_at_hand if 'J1' in x][0]
            player.cards_at_hand.remove(card)
            # key = (self.board_positions[c_row][c_col], c_row, c_col)
            # (pic, (x_value, y_value)) = self.board_pics[key]
            if self.logging:
                print('[+] Executed move to remove', self.board_positions[move[0][0]][move[0][1]])

            # self.screen.blit(pic, (x_value, y_value))
            # pygame.display.update()
        if move[1] == '0posJ2':
            self.coin_positions[move[0][0]][move[0][1]] = player.team[1]
            card = [x for x in player.cards_at_hand if 'J2' in x][0]
            player.cards_at_hand.remove(card)
            # key = (self.board_positions[move[0][0]][move[0][1]], move[0][0],move[0][1])
            # x_value, y_value = self.board_pics[key][1]
            if self.logging:
                print('[+] Executed move to put with Joker', self.board_positions[move[0][0]][move[0][1]])

            # self.screen.blit(self.get_coin_picture(player.team),
            #                   (x_value + self.coin_size[0] / 2, y_value + self.coin_size[1] / 2))
            # pygame.display.update()
        return card