import numpy as np
from numpy.random import randint as rdi
import graphic as gp
import time
import sys

class Game(object) :
    #####################################QUEUE#################################
    class queue(object) :
        def __init__(self,arg_height,arg_cube_size,arg_cube_step) :
            self.height = arg_height
            self.cube_size = arg_cube_size
            self.capacity = 10000
            self.cube_step = arg_cube_step
            self.frt = 0
            self.end = -1
            self.qu = np.zeros((self.capacity,2))
            self.delta = 0
            
        def add(self,col) :
            self.end = (self.end+1)%self.capacity
            self.qu[self.end] = np.array([self.delta+self.cube_size,col])
            
        def update(self) :
            self.delta+=self.cube_step
            while (not(self.empty()) and self.delta-self.qu[self.frt,0]>self.height+self.cube_size):
                self.frt = (self.frt+1)%self.capacity
                
        def get(self,i) :
            return np.array([   self.delta-self.qu[(self.frt+i)%self.capacity,0]-self.cube_size+1,
                                self.qu[(self.frt+i)%self.capacity,1]-self.cube_size+1,
                                self.delta-self.qu[(self.frt+i)%self.capacity,0],
                                self.qu[(self.frt+i)%self.capacity,1]
                            ]).astype(int)
        def empty(self) :
            return self.frt==self.end+1
            
        def size(self) :
            if (self.empty()) :
                return 0
            if (self.frt<=self.end) :
                return (self.end-self.frt+1)
            if (self.end<self.frt) :
                return (self.end-self.frt+self.capacity+1)

        def reset(self) :
            self.frt = 0
            self.end = -1
            self.delta = 0
            
    #################################GAME######################################
    def __init__(self) :
        #--------------------------CONSTANT------------------------------------
        self.panel_color = np.array([255,1,1])
        self.background_color = np.array([255,255,255])
        self.cube_color = np.array([1,1,1])

        self.height = 64
        self.width = 64
        self.actions = 2

        self.panel_height0 = 58
        self.panel_height1 = 60

        self.panel_width = 12
        self.cube_size = 4
        
        self.cube_step = 4
        self.panel_step = 4

        self.frame_per_cube = 2  #spawning rate
        self.cube_speed = 1      #frame per 1 pixel movement 
        self.panel_speed = 1    #frame per 1 pixel movement
        
        self.penalty = -100
        
        self.init_state = np.zeros((self.height,self.width,3))
        self.init_state[:,:] = np.copy(self.background_color)
        self.init_state[self.panel_height0:self.panel_height1+1,self.height/2-self.panel_width/2:self.width/2+self.panel_width/2]=np.copy(self.panel_color)
        #------------------------CURRENT-STATE---------------------------------
        self.current_state = np.copy(self.init_state)
        self.panel_pos = self.width/2-self.panel_width/2        
        self.cubes = self.queue(self.height,self.cube_size,self.cube_step)
        self.frame_index = 0
        self.game_over = False
        
    def position(self) :
        return np.array([self.panel_height0,self.panel_pos,self.panel_height1,self.panel_pos+self.panel_width-1])
    
    def reset(self) :
        self.current_state = np.copy(self.init_state)
        self.panel_pos = self.width/2-self.panel_width/2        
        self.cubes.reset()
        self.frame_index = 0
        self.game_over = False
    
    def new_cube(self) :
        l = []
        cnt = np.zeros(self.width+1)
        for i in range(self.cubes.size()-1,-1,-1) :
            coor = self.cubes.get(i)
            if (coor[0]>-self.cube_size+2) :
                break
            coor = self.cubes.get(i)
            cnt[max(0,coor[1]-1)]+=1
            cnt[min(self.width,coor[3]+self.cube_size+2)]-=1
        if (cnt[0]==0) :
            l.append(0)
        for i in range(1,self.width) :
            cnt[i]+=cnt[i-1]
            if (cnt[i]==0) :
                l.append(i)
        if (len(l)>0) :
            x = rdi(0,len(l))
            self.cubes.add(l[x])
        
    def collide(self,a) :
        minx = max(a[0,0],a[1,0])
        miny = max(a[0,1],a[1,1])
        maxx = min(a[0,2],a[1,2])
        maxy = min(a[0,3],a[1,3])
        return (minx<maxx and miny<maxy)
    
    def next_frame(self,action,fast=True) :
        if (self.game_over) :
            return [0,self.penalty,self.current_state]
        if (self.frame_index%self.frame_per_cube==0) :
            self.new_cube()
        if (fast) :
            #Update panel
            if (self.frame_index%self.panel_speed==0) :
                if (action==0 and self.panel_pos>0) :
                    self.panel_pos=max(0,self.panel_pos-self.panel_step)
                    self.current_state[self.panel_height0:self.panel_height1+1,
                                       self.panel_pos:self.panel_pos+self.panel_step]=np.copy(self.panel_color)
                    self.current_state[self.panel_height0:self.panel_height1+1,
                                       self.panel_pos+self.panel_width:min(self.width,self.panel_pos+self.panel_width+self.panel_step)]=np.copy(self.background_color)
                if (action==1 and self.panel_pos+self.panel_width<self.width) :
                    self.panel_pos=min(self.width-self.panel_width,self.panel_pos+self.panel_step)
                    self.current_state[self.panel_height0:self.panel_height1+1,
                                       self.panel_pos+self.panel_width-self.panel_step:min(self.width,self.panel_pos+self.panel_width)]=np.copy(self.panel_color)
                    self.current_state[self.panel_height0:self.panel_height1+1,
                                       max(0,self.panel_pos-self.panel_step):self.panel_pos]=np.copy(self.background_color)
            #Update cubes        
            if (self.frame_index%self.cube_speed==0) :
                self.cubes.update()  
                for i in range(self.cubes.size()) :
                    cube = self.cubes.get(i)
                    if (cube[0]-1) in range(0,self.height+self.cube_size) :
                        self.current_state[max(0,cube[0]-self.cube_step):cube[0],cube[1]:cube[3]+1]=np.copy(self.background_color)
                    if cube[2]-self.cube_step+1 in range(0,self.height) :
                        self.current_state[cube[2]-self.cube_step+1:min(self.height,cube[2]+1),cube[1]:cube[3]+1]=np.copy(self.cube_color)
        else :
            self.current_state[:,:] = np.copy(self.background_color)
            #Update panel
            if (self.frame_index%self.panel_speed==0) :
                if (action==0 and self.panel_pos>0) :
                    self.panel_pos=max(0,self.panel_pos-self.panel_step)
                if (action==1 and self.panel_pos+self.panel_width<self.width) :
                    self.panel_pos=min(self.width-self.panel_width,self.panel_pos+self.panel_step)
            coor = self.position()
            self.current_state[coor[0]:coor[2]+1,coor[1]:coor[3]+1] = np.copy(self.panel_color)
            #Update cubes        
            if (self.frame_index%self.cube_speed==0) :
                self.cubes.update()
                for i in range(self.cubes.size()) :
                    cube = self.cubes.get(i)
                    self.current_state[cube[0]:cube[2]+1,cube[1]:cube[3]+1] = np.copy(self.cube_color)
        #Check collision
        for i in range(self.cubes.size()) :
            cube = self.cubes.get(i)
            if (cube[2]<=self.panel_height0) :
                break
            if (self.collide(np.array([cube,self.position()]))) :
                self.game_over = True
                return [0,self.penalty,self.current_state]
        self.frame_index+=1
        return [1,1,self.current_state]

#------------------------------TESTING-ONLY------------------------------------
if __name__=='__main__' :
    start = time.time()
    game = Game()
    for i in range(9500) :
        state = game.next_frame(rdi(0,game.actions))
        img = state[2]
        if (state[0]==0) :
           game.reset()
        else :
           print "---"
           img = gp.transform(state[2])
           gp.show_grey(img)
#           img = state[2]
#           gp.show_rgb(img)
           pass
    print "{0:.3f}s".format(time.time()-start)