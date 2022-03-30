#
from basicenv import *
def sequence():
    no_players = 2
    players = [Player(i, (0, i)) for i in range(1, no_players+1)]
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
    if each_player_reward[0] > 0:
        print("Player 1 wins")
        env.display_board()
        return 1
    elif each_player_reward[1] > 0:
        print("Player 2 wins")
        env.display_board()
        return 2
    else:
        print("Draw")
        env.display_board()
        return 0





# game_instance.mainloop()
print(sequence())