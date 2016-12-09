# Deep Q Learning
Deep Q Learning - Convolutional neural network  
Used libraries : Theano - Keras, Numpy and Pygame  

The agent is a red panel which can move left and right trying to avoid collision with other black bricks which are falling from the sky.  
Given images of the current state of the game, the agent can neatly choose the best move to survive as long as possible.  

Algorithm :  
  Inspired by DeepMind's reinforcement learning with Atari game.  
  
		Collision penalty : -100  
		Reward for survival after one frame : 1  
		Future discount : 70%  

  The agent is a convolutional neural network with following struture :  

    Input shape :       (4,32,32) -- 4 lastest gray-color 32x32 frames of the current game.  
    Convolutional 2D :  (16,4,4)  -- 16 filters with size 4x4  
    RELU layer :        max(0,x)  -- Apply x = max(0,x) for the output of previous layer's neurons   
    Convolutional 2D :  (32,4,4)  -- 32 filters with size 4x4  
    RELU layer  
    Fully connected layer with 128 output units.  
    RELU layer  
    Fully connected layer with 2 output units -- expected rewards of moving left and right  
  
  Training :  
	
    Lossing function : mean squared error  
    Backpropagation method : rmsprop  
    Learning rate : 2e-5  
    Episode : about 2600  
  
  Modification :  
    Since the number of penalty states is very small compared with bonus states, a filter condition was added.  
    
    Whenever the current state of the game is add to the experience replay,  
      If the current state is a penalty state :  
        Add the current state to the experience replay with probability 100%.  
      Else :  
        Add the current state to the experience replay with probability 50%.
Result :  
![alt tag](https://github.com/pqhuy98/Deep-Q-Learning/blob/master/reinforcement-learning.gif)


Reference :  
  https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf  
