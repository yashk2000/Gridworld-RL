# Gridworld-RL

## Using a DQN with experience relay and a target netwwork to get the best results

**Experience Relay**: 

Catastrophic forgetting happens when two game states are very similar and yet lead to very different outcomes, the Q function will get “confused” and won’t be able to learn what to do. In order to overcome this, we use experience relay. 

Here’s how experience replay works:

1) In state s, take action a, and observe the new state `st+1` and reward `rt+1`.

2) Store this as a tuple (`s, a, st+1 , rt+1`) in a list.

3) Continue to store each experience in this list until you have filled the list to a specific length (this is up to you to define).

4) Once the experience replay memory is filled, randomly select a subset (again, you need to define the subset size).

5) Iterate through this subset and calculate value updates for each subset; store these in a target array (such as Y) and store the state, s, of each memory in X.

6) Use X and Y as a mini-batch for batch training. For subsequent epochs where the array is full, just overwrite old values in your experience replay memory array.

Thus, in addition to learning the action value for the action we took, we’re also going to use a random sample of past experiences to train on, to prevent catastrophic forgetting.

![2021-03-17_18-31](https://user-images.githubusercontent.com/41234408/111471650-205c0600-874f-11eb-87b8-fe828a496239.png)

**Target Learning**

Since the rewards may be sparse (we only give a significant reward upon winning or losing the game), updating on every single step, where most steps don’t get any significant reward, may cause the algorithm to start behaving erratically, which is called learning instability. 
To mitigate this, we use target learning. 

Let’s run through the sequence of events again, with the target network in play:

1) Initialize the Q-network with parameters (weights) `θQ`.

2) Initialize the target network as a copy of the Q-network, but with separate parameters `θT`, and set `θT = θQ`.

3) Use the epsilon-greedy strategy with the Q-network’s Q values to select action a.

4) Observe the reward and new state `rt+1` ,`st+1`.

5) The target network’s Q value will be set to `rt+1` if the episode has just been terminated (i.e., the game was won or lost) or to `rt+1 + γmaxQθR(st+1)` otherwise (notice the use of the target network here).

6) Backpropagate the target network’s Q value through the Q-network (not the target network).

7) Every C number of iterations, set `θT = θQ` (i.e., set the target network’s parameters equal to the Q-network’s parameters).

The idea is that we update the main Q-network’s parameters on each training iteration, but we decrease the effect that recent updates have on the action selection, hopefully improving stability.

![2021-03-17_18-35](https://user-images.githubusercontent.com/41234408/111472036-95c7d680-874f-11eb-8cc3-4860e9a1277c.png)
