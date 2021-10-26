
class Agent():
    def __init__(self):
        self.obs = []
        self.cards_at_hand = []
    def get_cards_at_start(self,cards):
        for i in cards:
            self.cards_at_hand.append(i)

    def get_obs(self,obs):
        self.obs.append(obs)

    #action
    #place card based on reward
    #