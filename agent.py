# Base Policy Agent
import math
import numpy as np
from basicenv import *
import itertools
import torch
from neuralnet import NeuralNet
"""
Create a simulator
Create a belief state generator
For each belief state run the simulator
"""


class BeliefStateGenerator:
    def __init__(self, board, discarded_cards, curr_player_hand, sampling_no):
        self.board = board
        self.discarded_cards = discarded_cards
        self.curr_player_hand = curr_player_hand
        self.deck = Deck()
        self.cards_for_opponent = []
        self.sampling_no = sampling_no

    def create_belief_state(self):
        """
        Create a belief state generator
        """
        curr_player_hand = self.curr_player_hand
        for i in curr_player_hand:
            self.deck.card_deck.remove(i)
        for i in self.discarded_cards:
            self.deck.card_deck.remove(i)
        # from self.deck.card_deck sample combinations of 5 cards and store them in a list
        self.cards_for_opponent = [random.sample(self.deck.card_deck, 5) for _ in range(self.sampling_no)]
        belief_states = []
        for i in self.cards_for_opponent:
            belief_states.append((deepcopy(self.board), deepcopy(self.deck), curr_player_hand, i))
        return belief_states


"""
New player class
makemove function will take board and discarded cards as input
create belief states
run simulator for each belief state
"""
class NeuralNetPlayer:
    def __init__(self, id, team, train=False):
        self.name = 'Neural Net Player ' + str(id)
        self.id = id
        self.cards_at_hand = []
        self.team = team
        self.no_of_sequences = []
        self.training_samples = []
        self.sampling_number = 50
        self.nn = NeuralNet()
        self.train = train


    def get_cards(self):
        for suit in ['C', 'S', 'H', 'D']:
            for value in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J1', 'J2', 'Q', 'K']:
                if value == 'J2' and (suit == 'S' or suit == 'H'):
                    continue
                if value == 'J1' and (suit == 'C' or suit == 'D'):
                    continue
                self.cards_for_move_conversion.append(value + suit)

    def make_move(self, board, discarded_cards):
        """
        Make move function will take board and discarded cards as input
        create belief states
        run simulator for each belief state
        """
        if self.train:
            print("------------------------------Generating new belief states--------------------------------")
            belief_state_generator = BeliefStateGenerator(deepcopy(board), discarded_cards, deepcopy(self.cards_at_hand), self.sampling_number)
            belief_states = belief_state_generator.create_belief_state()
            try:
                self.nn.load(filename='temp.pth.tar')
            except:
                print('No model found')
            initial = True
            for belief_state in belief_states:
                print("------------------------------Iteration: ", belief_states.index(belief_state))
                simulator = Simulator(deepcopy(belief_state), deepcopy(discarded_cards), initial=initial)
                # self.training_samples.append()
                examples, _ = simulator.simulate()
                self.training_samples+=examples
                if len(self.training_samples) > self.nn.nnet.batch_size:
                    if len(self.training_samples) > self.nn.nnet.batch_size*2:
                        self.training_samples.pop(0)
                    random.shuffle(self.training_samples)
                    self.nn.train(self.training_samples)
                    self.nn.save(filename='temp.pth.tar')
                    initial = False
            self.nn.save(filename='best.pth.tar')

            return self.get_move_from_nn(board, discarded_cards)
        else:
            self.nn.load(filename='best.pth.tar')
            return self.get_move_from_nn(board, discarded_cards)


    def get_move_from_nn(self, board, discarded_cards):
        belief_state_generator2 = BeliefStateGenerator(deepcopy(board), discarded_cards, deepcopy(self.cards_at_hand),
                                                      self.sampling_number)
        belief_states2 = belief_state_generator2.create_belief_state()
        moves_dict = {}
        for belief_state2 in belief_states2:
            simulator2 = Simulator(deepcopy(belief_state2), deepcopy(discarded_cards))
            valid_moves = simulator2.env.board.get_legal_moves(simulator2.players[0])
            for i in valid_moves:
                temp_simulator = deepcopy(simulator2)
                next_state = temp_simulator.get_next_state(i)
                pi, v = self.nn.predict(next_state)

                if i in moves_dict:
                    moves_dict[i] += v
                else:
                    moves_dict[i] = v
        max_move = max(moves_dict, key=moves_dict.get)
        return max_move



    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_player_pos(self, board):
        board_pos_player = []
        for i in range(10):
            for j in range(10):
                if (i, j) in [(0, 0), (0, 9), (9, 0), (9, 9)]:
                    continue
                if board.coin_positions[i][j] == self.team[1]:
                    board_pos_player.append((i, j))
        return board_pos_player

    def make_move_basic(self, board, discarded_cards=[]):
        moves = board.get_legal_moves(self)
        moves_put = [i for i in moves if i[1] != '1pos']
        moves_remove = [i for i in moves if i[1] == '1pos']
        # Get all the board positions of this player
        # Get all the board positions of the opponent
        board_pos_player = self.get_player_pos(board)
        if board_pos_player == []:
            return random.choice(moves)
        global_min = math.inf
        global_min_move = None
        for move in moves_put:
            pos1 = move[0]
            local_min = math.inf
            for pos2 in board_pos_player:
                local_min = min(local_min, self.manhattan_distance(pos1, pos2))
            if local_min < global_min:
                global_min = local_min
                global_min_move = move
        global_remove_min = math.inf
        global_remove_min_move = None
        for move in moves_remove:
            pos1 = move[0]
            local_min = math.inf
            for pos2 in board_pos_player:
                local_min = min(local_min, self.manhattan_distance(pos1, pos2))
            if local_min < global_remove_min:
                global_remove_min = local_min
                global_remove_min_move = move
        if global_min < global_remove_min:
            return global_min_move
        else:
            return global_remove_min_move


    def moves_to_pi_indices(self, moves):
        """
        Convert move to card
        """
        ravel_moves = []
        for move in moves:
            if move[1] == '0pos':
                ravel_moves.append(np.ravel_multi_index(move[0], (10,10)))
            elif move[1] == '1pos':
                ravel_moves.append(100+np.ravel_multi_index(move[0], (10,10)))
            elif move[1] == '0posJ2':
                ravel_moves.append(200+np.ravel_multi_index(move[0], (10,10)))
        return ravel_moves


    def pi_to_moves(self, index):
        # unravel_moves = []
        # for index,i in enumerate(pi):
        if index < 100:
            return (np.unravel_index(index, (10,10)), '0pos')
        elif index < 200:
            return (np.unravel_index(index-100, (10,10)), '1pos')
        else:
            return (np.unravel_index(index-200, (10,10)), '0posJ2')


    def validate_action(self, board, pi):
        """
        Mask invalid actions in pi
        """
        valid_moves = board.get_legal_moves(self)
        card_indices = self.moves_to_pi_indices(valid_moves)
        for i in range(len(pi)):
            if i not in card_indices:
                pi[i] = 0
        return pi




"""
Simulator class replicates the environment class
"""
class Simulator():
    def __init__(self, belief_state, discarded_cards, initial=False, logging=False):
        self.belief_state = belief_state
        self.board = belief_state[0]
        self.deck = belief_state[1]
        self.initialplay = initial
        self.opponent = BasePolicyAgent(2,(0,2))
        self.opponent.cards_at_hand = deepcopy(belief_state[3])
        curr_player = NeuralNetPlayer(1,(0,1))
        curr_player.cards_at_hand = deepcopy(belief_state[2])
        if not self.initialplay:
            curr_player.nn.load(filename='temp.pth.tar')
        self.players = [curr_player, self.opponent]
        self.env = Env(self.players, logging=logging, sim=True)
        self.env.discarded_cards = discarded_cards
        self.env.board = self.board
        self.env.deck = self.deck
        self.training_examples = []

    def get_next_state(self, action):
        """
        Returns the next state and reward after performing action
        """
        for ind, player in enumerate(self.players):
            if ind == 0:
                move = action
            else:
                move = player.make_move(self.env.board)
            _, reward, done, info = self.env.step(player, move)
        return self.get_state(self.players[0], self.players[1])



    def simulate(self):
        no_players = 2
        each_player_reward = [0 for i in range(no_players)]
        running = True
        while running:
            for ind, player in enumerate(self.players):
                self.env.board.change_corners_for_win_check(player)
                if ind == 0:
                    state = self.get_state(player, self.players[(ind+1)%2])
                    if not self.initialplay:
                        pi, v = player.nn.predict(state)
                        pi = player.validate_action(self.env.board, pi)
                        #pick max action in pi
                        action = np.argmax(pi)
                        if action in [0,9,90,99,100,109,199,200,209,299]:
                            print('Invalid move')
                            move = player.make_move_basic(self.env.board)
                            flag=1
                        else:
                            move = player.pi_to_moves(action)
                            flag=2
                        self.training_examples.append((state, ind, pi, None))
                        print(move, flag)
                        if move == None:
                            print('Invalid move')
                            print(self.env.board.get_legal_moves(player))

                    else:
                        move = player.make_move_basic(self.env.board)
                        # print("MOVE",move)
                        #Convert move to pi
                        pi_index = player.moves_to_pi_indices([move])
                        pi = np.zeros(300)
                        pi[pi_index] = 1
                        self.training_examples.append((state, ind, pi, None))

                else:
                    move = player.make_move(self.env.board)
                # print("PLAYER",ind,"MOVE",move)
                _, reward, done, info = self.env.step(player, move)
                # print("Recieved done",done)
                if done:
                    running = False
                    if reward != 1e-4:
                        each_player_reward[ind] = reward
                        each_player_reward[(ind + 1) % no_players] = -reward
                    else:
                        each_player_reward[ind] = 0
                        each_player_reward[(ind + 1) % no_players] = 0
                    break
        print("Reward",each_player_reward)
        return [(x[0],x[2],each_player_reward[x[1]]) for x in self.training_examples], each_player_reward[0]
    def get_state(self, player, opponent):
        """
        Get state of player
        """
        valid_moves = self.env.board.get_legal_moves(player)
        card_indices = player.moves_to_pi_indices(valid_moves)
        player_state = np.zeros(300)
        player_state[card_indices] = 1
        # player_state = np.reshape(player_state, (1,300))
        opponent_valid_moves = self.env.board.get_legal_moves(opponent)
        opponent_card_indices = player.moves_to_pi_indices(opponent_valid_moves)
        opponent_state = np.zeros(300)
        opponent_state[opponent_card_indices] = 1
        # opponent_state = np.reshape(opponent_state, (1,300))
        return  np.concatenate((np.array(deepcopy(self.env.board.coin_positions)).flatten(), player_state, opponent_state), axis=0)




class BasePolicyAgent(Player):
    def __init__(self, id, team):
        self.name = 'Base Policy Player ' + str(id)
        self.id = id
        self.cards_at_hand = []
        self.team = team
        self.no_of_sequences = []

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_player_pos(self, board):
        board_pos_player = []
        for i in range(10):
            for j in range(10):
                if (i, j) in [(0, 0), (0, 9), (9, 0), (9, 9)]:
                    continue
                if board.coin_positions[i][j] == self.team[1]:
                    board_pos_player.append((i, j))
        return board_pos_player

    def make_move(self, board, discarded_cards=[]):
        moves = board.get_legal_moves(self)
        moves_put = [i for i in moves if i[1] != '1pos']
        moves_remove = [i for i in moves if i[1] == '1pos']
        # Get all the board positions of this player
        # Get all the board positions of the opponent
        board_pos_player = self.get_player_pos(board)
        if board_pos_player == []:
            return random.choice(moves)
        global_min = math.inf
        global_min_move = None
        for move in moves_put:
            pos1 = move[0]
            local_min = math.inf
            for pos2 in board_pos_player:
                local_min = min(local_min, self.manhattan_distance(pos1, pos2))
            if local_min < global_min:
                global_min = local_min
                global_min_move = move
        global_remove_min = math.inf
        global_remove_min_move = None
        for move in moves_remove:
            pos1 = move[0]
            local_min = math.inf
            for pos2 in board_pos_player:
                local_min = min(local_min, self.manhattan_distance(pos1, pos2))
            if local_min < global_remove_min:
                global_remove_min = local_min
                global_remove_min_move = move
        if global_min < global_remove_min:
            return global_min_move
        else:
            return global_remove_min_move
