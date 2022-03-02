# from tkinter import *
import math
import random
import pygame
import time
import numpy as np
from copy import deepcopy
#1-blue
#2-red
#3-both
#TODO: Playing with Human
#   1. Display deck
#   2. Based on deck activate places to put tokens
#   3. Update Player object based on human actions
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
        self.screen=pygame.display.set_mode([self.board_size,self.board_size])
        self.board_positions = [['','2S','3S','4S','5S','6S','7S','8S','9S',''],['6C','5C','4C','3C','2C','AH','KH','QH','10H','10S'],['7C','AS','2D','3D','4D','5D','6D','7D','9H','QS'],['8C','KS','6C','5C','4C','3C','2C','8D','8H','KS'],['9C','QS','7C','6H','5H','4H','AH','9D','7H','AS'],['10C','10S','8C','7H','2H','3H','KH','10D','6H','2D'],['QC','9S','9C','8H','9H','10H','QH','QD','5H','3D'],['KC','8S','10C','QC','KC','AC','AD','KD','4H','4D'],['AC','7S','6S','5S','4S','3S','2S','2H','3H','5D'],['','AD','KD','QD','10D','9D','8D','7D','6D','']]
        self.board_pics = {}
        self.red_coin = (self.coin_size,1)
        self.blue_coin = (self.coin_size,2)
        self.coin_positions = [[0]*10 for _ in range(10)]
        # self.coin_positions[0][0],self.coin_positions[9][0],self.coin_positions[0][9],self.coin_positions[9][9],=3,3,3,3
        # self.initialize_board()



class Deck():
    def __init__(self):
        self.card_deck = []
        self.add_cards()
    def add_cards(self):
        for i in range(2):
            for suit in ['C', 'S', 'H', 'D']:
                for value in ['2','3','4','5','6','7','8','9','10','A','J1','J2','Q','K']:
                    if value=='J2' and (suit == 'S' or suit == 'H'):
                        continue
                    if value == 'J1' and (suit == 'C' or suit == 'D'):
                        continue
                    self.card_deck.append(value+suit)
        random.shuffle(self.card_deck)

class Player():
    def __init__(self,id,team,type):
        self.id = id
        self.cards_at_hand = []
        self.team = team
        self.type = type
        self.no_of_sequences = []

    # Make move function
    def find_positions(self,board):
        for ind_row,row in enumerate(board.board_positions):
            for ind_col,col in enumerate(row):
                if board.coin_positions[ind_row][ind_col]==0:
                    if col in self.cards_at_hand:
                        yield ((ind_row,ind_col),'0pos') #Will yield all positions where the player has cards to put the token
                    else:
                        yield ((ind_row,ind_col),'0posJ2') #Will yield all positions where the player needs J2 to put the token
                else:
                    if board.board_positions[ind_row][ind_col]!='' and board.coin_positions[ind_row][ind_col]!=self.team[1]:
                        yield ((ind_row,ind_col),'1pos')
    def get_coin_picture(self,team):
        if team[1]==1:
            return pygame.transform.scale(pygame.image.load('coins/red.png').convert(), team[0])
        else:
            return pygame.transform.scale(pygame.image.load('coins/blue.png').convert(),team[0])
    def put_without_j2(self,board,all_free_positions):
        c_row,c_col =  random.choice([position for position in all_free_positions if position[1]=='0pos'])[0]
        board.coin_positions[c_row][c_col]=self.team[1]
        self.cards_at_hand.remove(board.board_positions[c_row][c_col])
        key = (board.board_positions[c_row][c_col],c_row,c_col)
        x_value,y_value = board.board_pics[key][1]
        print("Putting on ", board.board_positions[c_row][c_col])
        picture = self.get_coin_picture(self.team)
        board.screen.blit(picture,(x_value+board.coin_size[0]/2,y_value+board.coin_size[1]/2))
        pygame.display.update()
    def put_with_j2(self,board,all_free_positions):
        c_row,c_col =  random.choice([position for position in all_free_positions if position[1]=='0pos' or position[1]=='0posJ2'])[0]
        board.coin_positions[c_row][c_col]=self.team[1]
        self.cards_at_hand.remove([x for x in self.cards_at_hand if 'J2' in x][0])
        key = (board.board_positions[c_row][c_col], c_row, c_col)
        x_value,y_value = board.board_pics[key][1]
        print("Putting (with J2) on ",board.board_positions[c_row][c_col])
        board.screen.blit(self.get_coin_picture(self.team),(x_value+board.coin_size[0]/2,y_value+board.coin_size[1]/2))
        pygame.display.update()
    def remove_with_j1(self,board,all_free_positions):
        c_row,c_col =  random.choice([position for position in all_free_positions if position[1]=='1pos'])[0]
        board.coin_positions[c_row][c_col]=0
        self.cards_at_hand.remove([x for x in self.cards_at_hand if 'J1' in x][0])
        key = (board.board_positions[c_row][c_col], c_row, c_col)
        (pic,(x_value,y_value)) = board.board_pics[key]
        print("Removing ", board.board_positions[c_row][c_col])
        board.screen.blit(pic,(x_value,y_value))
        pygame.display.update()
    def make_move(self,board):
        all_free_positions = [position for position in self.find_positions(board)]
        valid_actions = ['DoPut']
        if len([x for x in self.cards_at_hand if 'J2' in x]):
            valid_actions.append('DoJ2')
        if len([x for x in self.cards_at_hand if 'J1' in x]) and any(i[1]=='1pos' for i in all_free_positions):
            valid_actions.append('DoJ1')
        #Type of Player
        choice_of_action = random.choice(valid_actions)
        if choice_of_action=='DoPut':
            self.put_without_j2(board,all_free_positions)
        elif choice_of_action=='DoJ2':
            self.put_with_j2(board,all_free_positions)
        else:
            self.remove_with_j1(board,all_free_positions)


class Play():
    def __init__(self,players,board,cards_each):
        self.deck = Deck()
        self.board = board
        self.players = players
        self.no_of_cards = cards_each
        self.split_cards()
        self.winner = self.play()

    #Giving out cards to players
    def split_cards(self):
        for i in range(self.no_of_cards):
            for j in self.players:
                j.cards_at_hand.append(self.deck.card_deck.pop(0))

    #Check win condition
    def who_won(self,board,player):

        check_token1 = player.team[1]
        win_count =5
        transpose_coin_positions = np.array(deepcopy(board.coin_positions)).T.tolist()
        for i in range(10):
            for j in range(win_count):
                # If board has 5 in a row
                if all([token1==check_token1 for token1 in board.coin_positions[i][j:j+win_count]]):
                    print("ROW")
                    if(((i,j),(i,j+win_count-1)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i,j),(i,j+win_count-1)))
                    if len(player.no_of_sequences) == 2:
                        return False
                # If the board has 5 in a col
                if all([token1 == check_token1 for token1 in transpose_coin_positions[i][j:j + win_count]]):
                    print("Column")
                    if(((i,j),(i+win_count-1,j)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i,j),(i+win_count-1,j)))
                    if len(player.no_of_sequences) == 2:
                        return False
                # If the board has 5 diagonally
        i=0
        j=0
        while i<10:
            # next element
            count_diag = 0
            row, col = i, j
            while row < 10 and col > -1:
                if board.coin_positions[row][col] == check_token1:
                    count_diag += 1
                else:
                    count_diag = 0
                if count_diag == win_count:
                    print("Diag,left")
                    if (((i, j), (row, col)) not in player.no_of_sequences):
                        player.no_of_sequences.append(((i, j), (row, col)))
                    if len(player.no_of_sequences) == 2:
                        return False
                row += 1
                col -= 1
            count_diag = 0
            row, col = i, j
            while row < 10 and col < 10:
                if board.coin_positions[row][col] == check_token1:
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
                        return False
            if j<9:
                j+=1
            else:
                i+=1
        return True

    def change_corners_for_win_check(self,player):
        self.board.coin_positions[0][0],self.board.coin_positions[0][9],self.board.coin_positions[9][0],self.board.coin_positions[9][9] = [player.team[1]]*4

    #Play the game
    def play(self):
        for i in range(10):
            for j in range(10):
                if (self.board.board_positions[i][j] != ''):
                    pic = pygame.image.load('cards/' + self.board.board_positions[i][j] + '.png').convert()
                    pic = pygame.transform.scale(pic, self.board.pic_size)
                    # print(pic.get_rect())
                    x_value = j * self.board.board_size / 10
                    y_value = i * self.board.board_size / 10
                    self.board.board_pics[(self.board.board_positions[i][j],i,j)] = [pic, (x_value, y_value)]
                    self.board.screen.blit(pic, (x_value, y_value))
                    # self.board.screen.blit(coin_pic,(x_value+self.board.coin_size[0]/2,y_value+self.board.coin_size[1]/2))
                    # self.board.screen.blit(coin_pic,(x_value,y_value))
        winner = 0
        running = True
        while running:
            for player in self.players:
                #CHANGE THE CORNERS TO SELF.TEAM for winning check
                self.change_corners_for_win_check(player)
                print("PLAYER ",player.id," making a move.")
                # time.sleep(1)
                player.make_move(board=self.board)
                running = self.who_won(self.board,player)
                if running==False:
                    winner = player.id
                    print("Player ",player.id," WINS!!!")
                    pygame.image.save(self.board.screen, "winning.jpg")
                    break
                try:
                    add_card = self.deck.card_deck.pop(0)
                except:
                    print("GAME DRAW")
                    running=False
                    break
                player.cards_at_hand.append(add_card)


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Set the x, y postions of the mouse click
                    x, y = event.pos
                    print(x,y)
                    # for i in self.board.board_pics.values():
                    #     print(i[1][0],i[1][1])
                    #     if y in range(int(i[1][0]), int(i[1][0] + self.board.board_size * 0.0654)) and x in range(int(i[1][1]), int(i[1][1] + self.board.board_size // 10)):
                    #         print("Got it")
                    #         pygame.draw.circle(self.board.screen, (0,0,255), i[1][0] + int(self.board.board_size * 0.0654) // 2,
                    #                            i[1][1] + self.board.board_size // 20, 10)
            pygame.display.update()
        return winner


