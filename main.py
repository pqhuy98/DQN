import experience as xp
import game
import model
import numpy as np
import graphic as gp
from numpy.random import random
from numpy.random import randint as rdi

max_explore_rate = 0.40
min_explore_rate = 0.10
epoch = 1000000
height = 32
width = 32
channel = 4
xp_capacity = 1000
xp_nb_batch = 32
play_interval = 1
save_interval = 1000

game = game.Game()
net = model.CNN(height,width,channel,game.actions,"version01")
exp = xp.Experience(xp_capacity,game,channel,net)

for i in range(epoch) :
    print "Iteration",i
#    if (i%play_interval==play_interval-1) :
#        explore_rate = 0.0
#    else :
#        explore_rate = min_explore_rate+(epoch-i)*1.0/epoch*(max_explore_rate-min_explore_rate)
    explore_rate = 0.00
    step = 0
    while True :
        step+=1
        x = random()     
        if (x<explore_rate) :
            action = rdi(0,game.actions)
        else :
            current_frames = np.array([np.concatenate((exp.get_last(),[gp.transform(game.current_state,height,width)]))])
            action = net.choose_action(current_frames,i%play_interval==play_interval-1)[0]
        for j in range(1):#+(i%play_interval!=play_interval-1)) :
            exp.update()
            state = game.next_frame(action,False)
        exp_batch = exp.get(xp_nb_batch)
        if (i%play_interval==play_interval-1) :
            print "--"
            gp.show_rgb(state[2])
        else :
            if (exp_batch[0].shape[0]>0) :
                net.learn(exp_batch[0],exp_batch[1],(state[0]==0))
        if (state[0]==0) :
            game.reset()
            exp.reset()
            break
    print "step =",step
    if (i%save_interval==save_interval-1) :
        net.save("version01")
        print "Model saved"