import numpy as np
import game
import copy
import graphic as gp
from numpy.random import random

class Experience(object) :
    def __init__(self,arg_capacity,_game,arg_channel,arg_net) :
        self.capacity = arg_capacity
        self.game = _game
        self.net = arg_net
        self.channel = arg_channel
        self.imgs = np.zeros((self.capacity,self.channel+self.game.actions,self.net.height,self.net.width))
        self.reward = np.zeros((self.capacity,self.game.actions))
        self.size = 0
        self.full = False
        self.last = np.zeros((arg_channel-1,self.net.height,self.net.width))
        self.lpos = 0
        self.reset()
        
    def reset(self) :
        self.lpos = 0
        for i in range(self.channel-1):
            self.add_last(self.game.current_state)
            self.game.next_frame(1)
    
    def add_last(self,state) :
        self.last[self.lpos] = gp.transform(state,self.net.height,self.net.width)
        self.lpos = (self.lpos+1)%(self.channel-1)
        
    def add(self,imgs,rw) :
        self.imgs[self.size] = np.copy(imgs)
        self.reward[self.size] = np.copy(rw)
        self.size = (self.size + 1) % self.capacity
        if (self.size==0) :
            self.full = True
    
    def get(self,cnt) :
        size = self.size
        if (self.full) :
            size = self.capacity
        idx = np.random.choice(size,min(size,cnt),False)
        return [self.imgs[idx],self.reward[idx]]
    
    def get_last(self) :
        if (self.lpos==self.channel) :
            return np.copy(self.last[1:self.channel-1])
        else :
            return np.concatenate([
                self.last[(self.lpos)%(self.channel-1):(self.channel-1)],
                self.last[0:self.lpos]
            ])
    
    def update(self) :
        imgs = self.get_last().tolist()
        imgs.append(gp.transform(self.game.current_state))
        self.add_last(self.game.current_state)
        rew = []
        for i in range(self.game.actions) :
            _game = copy.deepcopy(self.game)       
            state = _game.next_frame(i)            
            imgs.append(gp.transform(state[2],self.net.height,self.net.width))
            rew.append(state[1])
        imgs = np.array(imgs)
        rew = np.array(rew)
        if (np.sum(rew)<0.5 or(self.size==0 and self.full==False)) :
            prob = 1
        else :
            prob = 0.5
        if (random()<prob) :
            self.add(imgs,np.array(rew));