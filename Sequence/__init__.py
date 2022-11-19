import numpy as np
import random
import time
import sys
import os
import math
from copy import deepcopy
import pygame
import itertools
from Sequence.board import Board
from Sequence.deck import Deck


"""
Initiliazing the environment with two players
player 1 will play

"""


class Env():
    def __init__(self, players, logging=False,sim=False):
        self.board = Board(False)
        self.logging = logging
        self.players = players
        self.deck = Deck()
        self.discarded_cards = []
        self.no_of_cards = 5
        self.action_space = []
        self.observation_space = []
        self.board_pics = {}
        if not sim:
            self.split_cards()

    def step(self, player, action, game_over=False):
        """

        :param player:
        :param action:
        :return: observation, reward, done, info
        """
        if game_over:
            return self.board.coin_positions, -1, True, {}
        card = self.board.execute_move(player, action)
        self.discarded_cards.append(card)
        try:
            player.cards_at_hand.append(self.deck.card_deck.pop(0))
            # print("Appended card to player's hand")
        except IndexError:
            if self.logging:
                print("Reshuffling discarded cards in deck")
                print(self.discarded_cards)
            self.deck.card_deck = self.discarded_cards.copy()
            random.shuffle(self.deck.card_deck)
            self.discarded_cards = []
            player.cards_at_hand.append(self.deck.card_deck.pop(0))
        done = self.board.is_win(player)
        if done:
            return self.board.coin_positions, 1, True, {}
        elif len(player.get_legal_moves(self.board)):
            return self.board.coin_positions, 0, False, {}
        else:
            return self.board.coin_positions, 1e-4, True, {}

    def reset(self, players):
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
            return pygame.transform.scale(pygame.image.load('coins/red.png').convert_alpha(), team[0])
        else:
            return pygame.transform.scale(pygame.image.load('coins/blue.png').convert_alpha(), team[0])

    def display_board(self, title='winning.jpg'):
        self.screen = pygame.display.set_mode([self.board.board_size, self.board.board_size])
        for i in range(10):
            for j in range(10):
                if (self.board.board_positions[i][j] != ''):
                    pic = pygame.image.load('cards/' + self.board.board_positions[i][j] + '.png').convert()
                    pic = pygame.transform.scale(pic, self.board.pic_size)
                    # print(pic.get_rect())
                    x_value = j * self.board.board_size / 10
                    y_value = i * self.board.board_size / 10
                    self.board_pics[(self.board.board_positions[i][j], i, j)] = [pic, (x_value, y_value)]
                    self.screen.blit(pic, (x_value, y_value))
                    # self.board.screen.blit(coin_pic,(x_value+self.board.coin_size[0]/2,y_value+self.board.coin_size[1]/2))
                    # self.board.screen.blit(coin_pic,(x_value,y_value))
                    if self.board.coin_positions[i][j] != 0:
                        key = (self.board.board_positions[i][j], i, j)
                        x_value, y_value = self.board_pics[key][1]
                        # print("Putting (with J2) on ", self.board_positions[i][j])
                        if self.board.coin_positions[i][j] == 1:
                            team = self.board.red_coin
                        else:
                            team = self.board.blue_coin
                        self.screen.blit(self.get_coin_picture(team),
                                          (x_value + self.board.coin_size[0] / 2, y_value + self.board.coin_size[1] / 2))
                        pygame.display.update()
        pygame.image.save(self.screen, title)

    def get_belief_state(self, player):
        curr_player_hand = player.cards_at_hand
        cards_for_opponent = []
        # from self.deck.card_deck create all combinations of 5 cards
        for i in itertools.combinations(self.deck.card_deck, self.no_of_cards):
            cards_for_opponent.append(i)
        print(len(cards_for_opponent))
        belief_states = []
        for i in cards_for_opponent:
            opponent_id = (player.id % 2) + 1
            belief_states.append((self.board, curr_player_hand, i))

        return belief_states


# A simulator for the game


