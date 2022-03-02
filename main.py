#
from env import *
def sequence():
    no_players = 2
    board = Board()
    players = [Player(i,board.blue_coin if i%2==0 else board.red_coin,'random') for i in range(1,no_players+1)]
    play1 = Play(players,board,5)
    return play1.winner

# game_instance.mainloop()
count_1 = 0
count_2 = 0
for i in range(100):
    winner = sequence()
    print(winner)
    if winner ==1:
        count_1+=1
    else:
        count_2+=1
print(count_1,count_2)