# Reinforcement Learning
Reinforcing Your Learning of Reinforcement Learning

这个是我在学习强化学习的过程中的一些记录，以及写的一些代码。建立这个Github项目主要是可以和大家一起相互学习和交流，也同时方便其他人寻找强化学习方面的资料。我为什么学习强化学习，主要是想把 AlphaZero 的那套方法（结合深度学习的蒙特卡洛树搜索）用在 RNA 分子结构预测上，但是具体怎么落地现在我也不知道。

首先看的书是 Richard S. Sutton 和 Andrew G. Barto 的 [Reinforcement Learning: An Introduction (Second edition)](http://incompleteideas.net/book/bookdraft2017nov5.pdf)。

看书的同时，也根据网上的一些文章写一些简单的代码，依次如下。

## Q-Learning

**Bellman equation:**
![Bellman equation](imgs/Bellman_equation.png)

**Frozen Lake Game**
<div align=center>
	<img width="100" height="100" src="imgs/frozenlake.png" alt="Frozen Lake Game">
</div>
基于 `Q-Learning` 玩 Frozen Lake 游戏：[code](blob/master/QLearning/QLearning_FrozenLake.py)

**Tic Tac Toe**
<div align=center>
	<img width="40" height="40" src="imgs/tic1.png" alt="Tic Tac Toe">
	<img width="40" height="40" src="imgs/tic2.png" alt="Tic Tac Toe">
	<img width="40" height="40" src="imgs/tic3.png" alt="Tic Tac Toe">
	<img width="40" height="40" src="imgs/tic4.png" alt="Tic Tac Toe">
	<img width="40" height="40" src="imgs/tic5.png" alt="Tic Tac Toe">
	<img width="40" height="40" src="imgs/tic6.png" alt="Tic Tac Toe">
	<img width="40" height="40" src="imgs/tic7.png" alt="Tic Tac Toe">
</div>

基于 `Q-Learning` 玩井字棋游戏：[code](blob/master/QLearning/QLearning_TicTacToe.py)

训练结果：
```
Q-Learning Player vs Q-Learning Player
====================
Train result - 100000 episodes
Q-Learning win rate: 0.45383
Q-Learning win rate: 0.3527
players draw rate: 0.19347
====================

Q-Learning Player vs Random Player
====================
Train result - 100000 episodes
Q-Learning win rate: 0.874
Random win rate: 0.03072
players draw rate: 0.09528
====================
```

[0]. [Diving deeper into Reinforcement Learning with Q-Learning](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe)
[1]. [Q* Learning with FrozenLake - Notebook](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Q%20Learning%20with%20FrozenLake.ipynb)


## Deep Q-Learning Network
<div align=center>
	<img width="300" height="300" src="imgs/DQN.png" alt="Deep Q-Learning Network">
</div>

**weights updation:**

![](imgs/DQN_loss.png)

**Play Doom Game**

![play Doom](imgs/play_doom.gif)

游戏环境这里使用的是[ViZDoom](http://vizdoom.cs.put.edu.pl)。神经网络是三层的卷积网络。[code](blob/master/DQN_Doom/DQN_Doom.py)

![DQN neural network](imgs/DQN_neural_network.png)

训练大约 1200 轮后结果如下：

![Doom loss](imgs/doom_loss.png)

```
Episode 0 Score: 61.0
Episode 1 Score: 68.0
Episode 2 Score: 51.0
Episode 3 Score: 62.0
Episode 4 Score: 56.0
Episode 5 Score: 33.0
Episode 6 Score: 86.0
Episode 7 Score: 57.0
Episode 8 Score: 88.0
Episode 9 Score: 61.0
[*] Average Score: 62.3
```

[0]. [An introduction to Deep Q-Learning: let’s play Doom](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)
[1]. [Deep Q learning with Doom - Notebook](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/DQN%20Doom/Deep%20Q%20learning%20with%20Doom.ipynb)

## Policy Gradients

<div align=center>
	<img src="imgs/policy_gradients.png" alt="Policy Gradients">
</div>

**Play CartPole Game**

<div align=center>
	<img src="imgs/play_cartpole.gif" alt="Play CartPole Game">
</div>

其中，Policy Gradient 神经网络如下图。[code](blob/master/Policy_Gradients/PG_Cartpole.py)

![Policy Gradient Network](imgs/pg_network.png)

训练大约 950 轮后结果如下：

![](imgs/pg_loss.png)
![](imgs/pg_mean_reward.png)

```
====================
Episode: 941
Reward: 39712.0
Mean Reward: 2246.384288747346
Max reward so far: 111837.0
====================
Episode: 942
Reward: 9417.0
Mean Reward: 2253.9883351007425
Max reward so far: 111837.0
====================
Episode: 943
Reward: 109958.0
Mean Reward: 2368.08156779661
Max reward so far: 111837.0
====================
Episode: 944
Reward: 73285.0
Mean Reward: 2443.125925925926
Max reward so far: 111837.0
====================
Episode: 945
Reward: 40370.0
Mean Reward: 2483.217758985201
Max reward so far: 111837.0
[*] Model Saved: ./model/model.ckpt
```

[0]. [An introduction to Policy Gradients with Cartpole and Doom](https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f)
[1]. [Cartpole: REINFORCE Monte Carlo Policy Gradients - Notebook](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb)
[2]. [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

## AlphaGoZero Introduction

这个是我通过阅读 AlphaGo Zero 的文献，以及结合网路上相关的一些文章，将这些内容通过自己的理解整合到这一个PPT中，用来在组会上简单的介绍 AlphaGo Zero 背后的方法和原理给同学和老师，同时也思考如何将其结合到其他领域。当然，其中也不仅仅包括 AlphaGo Zero 的内容，也有我最近看的另外一篇文章，他们的研究团队运用类似的方法来解魔方。[[pdf]](blob/master/AlphaGoZero_Intruduction/alphago_zero_introduction.pdf)

<div align=center>
	<img width="200" height="200" src="imgs/pdf_2.png" alt="pdf">
	<img width="200" height="200" src="imgs/pdf_3.png" alt="pdf">
</div>

[0]. [AlphaGo Zero - How and Why it Works](http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/)
[1]. [Alpha Go Zero Cheat Sheet](https://applied-data.science/static/main/res/alpha_go_zero_cheat_sheet.png)
[2]. [Mastering the game of Go with deep neural networks and tree search](https://deepmind.com/research/publications/mastering-game-go-deep-neural-networks-tree-search/)
[3]. [Mastering the game of Go without Human Knowledge](https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge/)


## Monte Carlo Tree Search

**Gomoku**
<div align=center>
	<img width="400" height="400" src="imgs/mcts_gomoku.png" alt="pdf">
</div>

MCTS vs Random Player [[code]](blob/master/MCTS/MCTS_Gomoku.py). Another MCTS on Tic Tac Toe [[code]](blob/master/MCTS/MCTS_TicTacToe.py).

[0]. [mcts.ai](http://mcts.ai/code/python.html)
[1]. [Introduction to Monte Carlo Tree Search](https://www.caktusgroup.com/blog/2015/09/24/introduction-monte-carlo-tree-search-1/)

## AlphaGomoku
使用AlphaGo Zero的方法实现的一个五子棋AI。

下图是自我博弈训练 3000 局棋后，与人类选手对局的结果，已经很难下赢了。
<div align=center>
	<img width="400" height="400" src="imgs/alphagomoku.png" alt="pdf">
</div>

策略估值网络提供了两个模型，分别是：
```
################
# Residual_CNN #
################

Network Diagram:
                |-----------------------|                  /---C---B---R---F---D---R---D---T [value head]
I---C---B---R---o---C---B---R---C---B---M---R--- ..... ---|
    \_______/     \_______________________/                \---C---B---R---F---D---S [polich head]
   [Conv layer]       [Residual layer]

I - input
B - BatchNormalization
R - Rectifier non-linearity, LeakyReLU
T - tanh
C - Conv2D
F - Flatten
D - Dense
M - merge, add
S - Softmax
O - output

##############
# Simple_CNN #
##############

Network Diagram:
                                               2(1x1)       64     1
    32(3x3)     64(3x3)    128(3x3)        /-----C-----F-----D-----D-----T  [value head]
I-----C-----R-----C-----R-----C-----R-----|
      \_____________________________/      \-----C-----F-----D-----S        [polich head]
           [Convolutional layer]               4(1x1)       w^2

I - input
B - BatchNormalization
R - ReLU
T - tanh
C - Conv2D
F - Flatten
D - Dense
S - Softmax
```

`8x8` 大小棋盘自我博弈训练 3000 局的结果如下：

![loss](AlphaGomoku/loss/Simple_CNN_8x8_loss.png)

```
[*] Episode: 2991, length: 42, start: O, winner: X, data: 336, time: 85s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.7491 - value_head_loss: 0.4658 - policy_head_loss: 1.0655
[*] Episode: 2992, length: 19, start: O, winner: O, data: 152, time: 40s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6507 - value_head_loss: 0.4631 - policy_head_loss: 0.9698
[*] Episode: 2993, length: 23, start: X, winner: X, data: 184, time: 47s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6409 - value_head_loss: 0.4322 - policy_head_loss: 0.9908
[*] Episode: 2994, length: 35, start: X, winner: X, data: 280, time: 71s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6128 - value_head_loss: 0.4528 - policy_head_loss: 0.9421
[*] Episode: 2995, length: 16, start: X, winner: O, data: 128, time: 35s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.7529 - value_head_loss: 0.4884 - policy_head_loss: 1.0466
[*] Episode: 2996, length: 22, start: O, winner: X, data: 176, time: 46s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6800 - value_head_loss: 0.4583 - policy_head_loss: 1.0038
[*] Episode: 2997, length: 16, start: X, winner: O, data: 128, time: 35s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6877 - value_head_loss: 0.4973 - policy_head_loss: 0.9725
[*] Episode: 2998, length: 22, start: X, winner: O, data: 176, time: 48s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6530 - value_head_loss: 0.4887 - policy_head_loss: 0.9464
[*] Episode: 2999, length: 16, start: X, winner: O, data: 128, time: 33s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6951 - value_head_loss: 0.4582 - policy_head_loss: 1.0189
[*] Episode: 3000, length: 9, start: X, winner: X, data: 72, time: 18s, win ratio: X 48.1%, O 48.5%, - 3.4%
Epoch 1/1
512/512 [==============================] - 1s 2ms/step - loss: 1.6760 - value_head_loss: 0.4743 - policy_head_loss: 0.9838
```

具体代码及训练好的模型参数请参考这里：[code](AlphaGomoku/)


[0]. [How to build your own AlphaZero AI using Python and Keras](https://applied-data.science/blog/how-to-build-your-own-alphazero-ai-using-python-and-keras/)
[1]. [Github: AppliedDataSciencePartners/DeepReinforcementLearning](https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning)
[2]. [Github: Rochester-NRT/RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo)
[3]. [28 天自制你的 AlphaGo (6) : 蒙特卡洛树搜索（MCTS）基础](https://zhuanlan.zhihu.com/p/25345778)
[4]. [AlphaZero实战：从零学下五子棋（附代码）](https://zhuanlan.zhihu.com/p/32089487)
[5]. [Github: junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
