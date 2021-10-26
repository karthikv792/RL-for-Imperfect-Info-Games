import random


class Board():
    def __init__(self):
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
        self.coin_positions = [[0]*10]*10
        self.coin_positions[0][0],self.coin_positions[9][0],self.coin_positions[0][9],self.coin_positions[9][9],=1,1,1,1

class Deck():
    def __init__(self):
        self.card_deck = []
    def add_cards(self):
        for i in range(2):
            for suit in ['C', 'S', 'H', 'D']:
                for value in ['1','2','3','4','5','6','7','8','9','10','A','J','Q','K']:
                    self.card_deck.append(value+suit)
        random.shuffle(self.card_deck)

class Player():
    def __init__(self,id):
        self.id = id
        self.cards_at_hand = []


class Play():
    def __init__(self,players,cards_each):
        self.deck = Deck()
        self.board = Board()
        self.players = [Player(i) for i in range(1,players+1)]
        self.no_of_cards = cards_each
        self.split_cards()

    def split_cards(self):
        for i in range(self.no_of_cards*len(self.players)):
            for j in self.players:
                j.cards_at_hand(self.deck.card_deck.pop(0))

    def play(self):
        while True:
            for i in self.players:
                i.make_move()


