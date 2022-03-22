# Mine Sweeper AI

In this project, I want to implement a minimalistic version of mine sweeper game and run some AI experiments on it.

+ Ideally, I want to follow OpenAI syntax and register it as a regular gym env.
+ After that, I like to implement a couple of RL algorithms and see how they would perform. For now, I am thinking of a simple policy gradient algorithm (REINFORCE) and a value based method (probably DQN). Maybe also a simple actor-critic one.
+ I like to test and see how different policy networks fair against one another, particularly seeing if convolutional networks can improve the performance in comparison to feed-forward ones, as taking actions in this game is a delicate choice.
+ I like to test a couple methods and see how I can create policies that are size-agnostic, or that can generalize well to different board sizes. (I am thinking of using GNNs or meta-RL)
+ I also like to implement some evolutionary methods and try to combine them with RL to see if I can get optimal **and** diverse policies.
+ Another interesting thing I want to test is to have one agent design the game board (place a fixed number of mines within the board) and another one to solve that task. This can be seen as a game between the two agents. Maybe the first agent can help the second one to learn more efficiently? Maybe we can formulate this as an adversarial game. I am not quite sure about this part. I will think about it more carefully in the future.
+ Some ideas about incorporating uncertainty about the location of mines are looming over my head. I may try to rigorously formalize them too.

