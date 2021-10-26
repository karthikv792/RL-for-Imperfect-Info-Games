# from tkinter import *
import math

import pygame
board_size = 800
print(int(board_size*0.0654))
class Board():
    def __init__(self):
        self.screen=pygame.display.set_mode([board_size,board_size])
        self.board_positions = [['','2S','3S','4S','5S','6S','7S','8S','9S',''],['6C','5C','4C','3C','2C','AH','KH','QH','10H','10S'],['7C','AS','2D','3D','4D','5D','6D','7D','9H','QS'],['8C','KS','6C','5C','4C','3C','2C','8D','8H','KS'],['9C','QS','7C','6H','5H','4H','AH','9D','7H','AS'],['10C','10S','8C','7H','2H','3H','KH','10D','6H','2D'],['QC','9S','9C','8H','9H','10H','QH','QD','5H','3D'],['KC','8S','10C','QC','KC','AC','AD','KD','4H','4D'],['AC','7S','6S','5S','4S','3S','2S','2H','3H','5D'],['','AD','KD','QD','10D','9D','8D','7D','6D','']]
        self.board_pics = {}
        self.initialize_board()

    def initialize_board(self):
        running = True
        while running:
            for i in range(10):
                for j in range(10):
                    if(self.board_positions[i][j]!=''):
                        pic = pygame.image.load('cards/'+self.board_positions[i][j]+'.png').convert()
                        pic = pygame.transform.scale(pic, (int(board_size*0.0654), board_size//10))
                        # print(pic.get_rect(),self.board_pics.values())
                        self.board_pics[self.board_positions[i][j]] = [pic,(j*board_size/10,i*board_size/10)]
                        self.screen.blit(pic,(j*board_size/10,i*board_size/10))
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

