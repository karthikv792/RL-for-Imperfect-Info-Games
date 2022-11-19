import random
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
