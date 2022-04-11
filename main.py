#
from basicenv import *
from agent import *
from matplotlib import pyplot as plt




def sequence(display=False):
    no_players = 2
    # players = [Player(i, (0, i)) for i in range(1, no_players+1)]
    players = [BasePolicyAgent(1, (0, 1)), BasePolicyAgent(2, (0, 2))]
    each_player_reward = [0 for i in range(no_players)]
    env = Env(players)
    running = True
    while running:
        for ind, player in enumerate(players):
            _, reward, done, info = env.step(player, player.make_move(env.board))
            if done:
                running = False
                if reward != 1e-4:
                    each_player_reward[ind] = reward
                    each_player_reward[(ind+1)%no_players] = -reward
                else:
                    each_player_reward[ind] = 0
                    each_player_reward[(ind+1)%no_players] = 0
                break
    if display:
        env.display_board()
    if each_player_reward[0] > 0:
        print("Player 1 wins")
        return 1, [player.name for player in players]
    elif each_player_reward[1] > 0:
        print("Player 2 wins")
        return 2, [player.name for player in players]
    else:
        print("Draw")
        return 0, [player.name for player in players]





# game_instance.mainloop()

wins = []
for i in range(10):
    winner, player_names = sequence()
    wins.append(winner)
# Count number of ones and twos in wins
one_count = wins.count(1)
two_count = wins.count(2)
three_count = wins.count(0)

# A bar chart of the number of wins for each player
plt.bar(player_names+["Draw"], [one_count, two_count, three_count], color=['red', 'blue', 'green'])
plt.show()


