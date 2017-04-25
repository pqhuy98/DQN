# Deep Q Learning
Deep Q Learning - Convolutional neural network  
Used libraries : Theano - Keras for GPU calculations and Numpy.

This project is about teaching computer to play game without having any preknowledge about the game (learn from raw image pixels). The algorithm was invented by Google DeepMind.  
In this sample experiment, the computer control a red panel which can move left or right try to avoid collision with other black bricks which fall from the sky. Given only gray-scaled images of the game and the rewards or penalties got after each move, the computer must learn a strategy to survive as long as possible.

Algorithm : Deep Q learning using convolutional neural network.  
Trained about 6 hours using Nvidia Geforce GTX 950M GPU.  

Result :  
![alt tag](https://github.com/pqhuy98/Deep-Q-Learning/blob/master/reinforcement-learning.gif)
  
Writted by <b>Pham Quang Huy</b>.
  
Reference :  
  https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
