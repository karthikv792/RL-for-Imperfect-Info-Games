# from tkinter import *
import math
import random
import pygame
#1-blue
#2-red
#3-both
"""
Things to consider
1. One eyed jack can remove
2. two eyed jacks can substitute anything
"""
class Board():
    def __init__(self):
        self.board_size = 800
        self.pic_size = (int(self.board_size*0.0654),self.board_size//10)
        self.coin_size = (int(self.board_size*0.0327),self.board_size//20)
        self.red_coin = pygame.transform.scale(pygame.image.load('coins/red.png').convert(), self.coin_size)
        self.blue_coin = pygame.transform.scale(pygame.image.load('coins/blue.png').convert(), self.coin_size)
        self.screen=pygame.display.set_mode([self.board_size,self.board_size])
        self.board_positions = [['','2S','3S','4S','5S','6S','7S','8S','9S',''],['6C','5C','4C','3C','2C','AH','KH','QH','10H','10S'],['7C','AS','2D','3D','4D','5D','6D','7D','9H','QS'],['8C','KS','6C','5C','4C','3C','2C','8D','8H','KS'],['9C','QS','7C','6H','5H','4H','AH','9D','7H','AS'],['10C','10S','8C','7H','2H','3H','KH','10D','6H','2D'],['QC','9S','9C','8H','9H','10H','QH','QD','5H','3D'],['KC','8S','10C','QC','KC','AC','AD','KD','4H','4D'],['AC','7S','6S','5S','4S','3S','2S','2H','3H','5D'],['','AD','KD','QD','10D','9D','8D','7D','6D','']]
        self.board_pics = {}
        self.coin_positions = [[0]*10]*10
        self.coin_positions[0][0],self.coin_positions[9][0],self.coin_positions[0][9],self.coin_positions[9][9],=3,3,3,3
        self.initialize_board()

    def initialize_board(self):
        running = True
        while running:
            for i in range(10):
                for j in range(10):
                    if(self.board_positions[i][j]!=''):
                        pic = pygame.image.load('cards/'+self.board_positions[i][j]+'.png').convert()
                        pic = pygame.transform.scale(pic, self.pic_size)
                        # print(pic.get_rect())
                        x_value = j*self.board_size/10
                        y_value = i*self.board_size/10
                        self.board_pics[self.board_positions[i][j]] = [pic,(x_value,y_value)]
                        self.screen.blit(pic,(x_value,y_value))
                        self.screen.blit(coin_pic,(x_value+self.coin_size[0]/2,y_value+self.coin_size[1]/2))
                        # self.screen.blit(coin_pic,(x_value,y_value))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Set the x, y postions of the mouse click
                    x, y = event.pos
                    print(x,y)
                    for i in self.board_pics.values():
                        print(i[1][0],i[1][1])
                        if y in range(int(i[1][0]),int(i[1][0] + board_size * 0.0654)) and x in range(int(i[1][1]),int(i[1][1] + board_size // 10)):
                            print("Got it")
                            pygame.draw.circle(self.screen, (0,0,255), i[1][0] + int(board_size * 0.0654) // 2,
                                               i[1][1] + board_size // 20, 10)
            pygame.display.update()

class Deck():
    def __init__(self):
        self.card_deck = []
    def add_cards(self):
        for i in range(2):
            for suit in ['C', 'S', 'H', 'D']:
                for value in ['1','2','3','4','5','6','7','8','9','10','A','J1','J2','Q','K']:
                    self.card_deck.append(value+suit)
        random.shuffle(self.card_deck)

class Player():
    def __init__(self,id,team):
        self.id = id
        self.cards_at_hand = []
        self.team = "blue" if team else "red"

    # Make move function
    def find_positions(self,board):
        for ind_row,row in enumerate(board.board_positions):
            for ind_col,col in enumerate(row):
                if board.coin_positions[ind_row][ind_col]==0:
                    if col in self.cards_at_hand:
                        yield ((ind_row,ind_col),'0pos')
                    else:
                        yield ((ind_row,ind_col),'0posJ2')

    def put_normally(self,board):
        all_free_positions = [position for position in self.find_postions(board)]
        return random.choice(all_free_positions)
    def put_using_j2(self,board):
        pass
    def make_move(self,board):
        
        #Actions you can take
        #Remove a piece
        #Put a piece 
            #1. Put using J2
            #2. Put normally
        pass
        

class Play():
    def __init__(self,players,cards_each):
        self.deck = Deck()
        self.board = Board()
        self.players = [Player(i,i%2==0) for i in range(1,players+1)]
        self.no_of_cards = cards_each
        self.split_cards()

    #Giving out cards to players
    def split_cards(self):
        for i in range(self.no_of_cards*len(self.players)):
            for j in self.players:
                j.cards_at_hand.append(self.deck.card_deck.pop(0))

    #Check win condition
    def who_won(self):
        #If board has 
        pass
    #Play the game
    def play(self):
        while True:
            for player in self.players:
                print("PLAYER ",player.id," making a move.")
                player.make_move(self.board)
