#
from Sequence import Env
from agents.RandomAgent import *
from agents.BasePolicyAgent import *
from agents.NeuralNetworkAgent import *

from matplotlib import pyplot as plt

total_wins_of_neural_net = []


def sequence(display=False, train=False, display_title="winning.jpg"):
    no_players = 2
    # players = [Player(i, (0, i)) for i in range(1, no_players+1)]
    players = [NeuralNetPlayer(1, (0, 1),  train=train), BasePolicyAgent(2, (0, 2))]
    each_player_reward = [0 for i in range(no_players)]
    env = Env(players)
    running = True
    iteration = 1
    while running:
        for ind, player in enumerate(players):
            _, reward, done, info = env.step(player, player.make_move(env.board, env.discarded_cards))
            if done:
                running = False
                if reward != 1e-4:
                    each_player_reward[ind] = reward
                    each_player_reward[(ind+1)%no_players] = -reward
                else:
                    each_player_reward[ind] = 0
                    each_player_reward[(ind+1)%no_players] = 0
                break
            # belief_states = env.get_belief_state(player)
        if train==True and iteration%500==0:
            wins = []
            for i in range(10):
                winner, player_names = sequence2(display=False, train=False)
                wins.append(winner)
            # Count number of ones and twos in wins
            one_count = wins.count(1)
            total_wins_of_neural_net.append(one_count)
            two_count = wins.count(2)
            three_count = wins.count(0)

            # A bar chart of the number of wins for each player
            plt.bar(player_names + ["Draw"], [one_count, two_count, three_count], color=['red', 'blue', 'green'])
            plt.show()
        iteration += 1

    if display:
        env.display_board(display_title)
    names = [player.name for player in players]
    del players
    del env
    if each_player_reward[0] > 0:
        print("Player 1 wins")
        return 1, names
    elif each_player_reward[1] > 0:
        print("Player 2 wins")
        return 2, names
    else:
        print("Draw")
        return 0, names


def sequence2(display=False, train=False, display_title="winning.jpg"):
    no_players = 2
    # players = [Player(i, (0, i)) for i in range(1, no_players+1)]
    players = [NeuralNetPlayer(1, (0, 1),train=train), BasePolicyAgent(2, (0, 2))]
    each_player_reward = [0 for i in range(no_players)]
    env2 = Env(players)
    running = True
    while running:
        for ind, player in enumerate(players):
            _, reward, done, info = env2.step(player, player.make_move(env2.board, env2.discarded_cards))
            if done:
                running = False
                if reward != 1e-4:
                    each_player_reward[ind] = reward
                    each_player_reward[(ind+1)%no_players] = -reward
                else:
                    each_player_reward[ind] = 0
                    each_player_reward[(ind+1)%no_players] = 0
                break
            # belief_states = env.get_belief_state(player)

    if display:
        env2.display_board(display_title)
    names = [player.name for player in players]
    del players
    del env2
    if each_player_reward[0] > 0:
        print("Player 1 wins")
        return 1, names
    elif each_player_reward[1] > 0:
        print("Player 2 wins")
        return 2, names
    else:
        print("Draw")
        return 0, names


# game_instance.mainloop()

wins = []
for i in range(50):
    winner, player_names = sequence(display=True, train=False)
    wins.append(winner)
# Count number of ones and twos in wins
print(wins)
one_count = wins.count(1)
two_count = wins.count(2)
three_count = wins.count(0)
print("Player 1 wins: ", one_count)
print("Player 2 wins: ", two_count)
print("Draw: ", three_count)
# A bar chart of the number of wins for each player
plt.bar(player_names+["Draw"], [one_count, two_count, three_count], color=['red', 'blue', 'green'])
plt.show()

#Plot a line graph with total wins of neural net
plt.plot(total_wins_of_neural_net)
plt.show()


