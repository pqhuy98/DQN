import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Convolution2D,Dense,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop
#import experience as xp
import game
import graphic as gp
import copy
from numpy.random import randint as rdi

class CNN(object) :
    def __init__(self,arg_height,arg_width,arg_channel,arg_actions,name="") :
        self.actions = arg_actions
        self.height = arg_height
        self.width = arg_width
        self.channel = arg_channel
        self.future_discount = 0.7
        self.NN = Sequential()
        if (name=="") :
            self.NN.add(Convolution2D(16,4,4,input_shape=(self.channel,self.height,self.width),border_mode='same'))
            self.NN.add(Activation('relu'))
#            self.NN.add(MaxPooling2D((2,2)))
            self.NN.add(Convolution2D(32,4,4,border_mode='same'))
            self.NN.add(Activation('relu'))
#            self.NN.add(MaxPooling2D((2,2)))
            self.NN.add(Flatten())
            self.NN.add(Dense(128))
            self.NN.add(Activation('relu'))
            self.NN.add(Dense(self.actions))
        else :
            self.load(name)
        self.NN.compile(loss='mean_squared_error',optimizer=RMSprop(lr=2e-5),metrics=['mean_squared_error']);
        
    def expected_reward(self,imgs) :
        #return np.reshape(np.max(self.NN.predict(imgs),axis=1),(-1,1))
        return np.max(self.NN.predict(imgs))
        
    def learn(self,exp_img,exp_reward,verbose=1) :
        X_train = exp_img[:,0:self.channel].astype('float32')
#        Y_train = (np.concatenate(
#            [self.expected_reward(np.concatenate((exp_img[:,1:self.channel],exp_img[:,[x]]),axis=1))\
#                for x in range(self.channel,self.channel+self.actions)],axis=1)*self.future_discount +
#            exp_reward).astype('float32')
        Y_train = np.copy(exp_reward)
        for i in range(exp_img.shape[0]) :
            for j in range(self.actions) :
                if (Y_train[i,j]>-0.5) :
                    Y_train[i,j]+=self.expected_reward(
                        np.array([
                        np.concatenate((exp_img[i,1:self.channel],exp_img[i,[self.channel+j]]))
                    ]))*self.future_discount
        Y_train = Y_train.astype('float32')
        
        self.NN.fit(X_train,Y_train,batch_size=50,nb_epoch=1,verbose=verbose)
#        print Y_train
#        Y_train = (np.concatenate(
#            [self.expected_reward(np.concatenate((exp_img[:,1:self.channel],exp_img[:,[x]]),axis=1))\
#                for x in range(self.channel,self.channel+self.actions)],axis=1)*self.future_discount +
#            exp_reward).astype('float32')
#        print Y_train
    
    def choose_action(self,a,verbose=False) :
        x = self.NN.predict(a)
        if verbose :
            print x
        return np.argmax(x,axis=1)
    
    def save(self,name) :
        self.NN.save(name+".h5")
    
    def load(self,name) :
        self.NN = load_model(name+".h5")
    
if __name__=="__main__" :
    net = CNN(32,32,2,3)
    channel = 4
    actions = 3
    game = game.Game()
    for i in range(10) :
        game.next_frame(rdi(0,3))
    exp = []
    for i in range(channel-1) :
        exp.append(gp.transform(game.next_frame(0)[2]))
    for i in range(actions) :
        tmp = copy.deepcopy(game)
        exp = np.concatenate((exp,[gp.transform(tmp.next_frame(i)[2])]))
    for i in range(exp.shape[0]) :
        if (i==channel-1) :
            print "---------------"
        gp.show_grey(exp[i])
    exp = np.array([exp])
    print exp
    rew = np.array([[0,-2,0]])
    for i in range(100) :
        net.learn(exp,rew)