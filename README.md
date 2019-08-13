# CartPole-v0

This is a solution to solve the OpenAI gym CartPole-v0 environment. For the initial development, I used two tutorials. These were as follows:

* [https://www.youtube.com/watch?v=ViwBAK8Hd7Q](https://www.youtube.com/watch?v=ViwBAK8Hd7Q)
* [https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/?completed=/q-learning-reinforcement-learning-python-tutorial/](https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/?completed=/q-learning-reinforcement-learning-python-tutorial/)

## The solution

1. Attributes

For my solution it uses the following attributes:

````

learning rate = 0.1
discount = 0.95
runs = 10000
reward when goal not reached = -375

````
In addition to this, the reward for a successful run is 1 but as this attribute cannot be adjusted I have left it out.

2. Bins and Q table

Before training can begin two things need to be set up. This first of these are bins which will represent the environment with discrete values instead of continuous values. This will lower the number of states that need to be adjusted. For CartPole-v0 the cart location, cart speed, pole and pole speed are recorded in the environment. Each of these has 20 bins and have been given a range for each bin. As OpenAI gives us the hax and min values I have hardcoded this in but there could be the case to discover optimal values with several runs before setting these.

The Q table also needs to be set up. This has the same shape as the bins with the addition axis for a number of actions that can be performed. For CartPole this is 2. Each value in the Q table is initially random.

To get an action from the Q table get_discrete_state() is used which will use the bins to get the index of the corresponding section from the Q table.

3. Training

For every run of the environment, the agent will try and keep the pole upright and within the bounds of the environment. It has the maximum of 200 moves to make at with point it has either passed (and will get the reward of 1) or it would have failed before reaching it and will thus get a reward of -375. For every move, there is a chance it will make a random move or will get the current best move from the Q table. Depending on the result on the environment this will either increase or decrease the value updated into the Q table under the current state and action used. The random moves start off occurring frequently but will decrease over time.

## Q learning

Q learning is a model-free reinforcement learning algorithm. It will learn a policy which will tell it what to do given a certain situation. Over the course of training, the Q learning will update its policy to find the optimal (or the closest it can get) action given a state.

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/47fa1e5cf8cf75996a777c11c7b9445dc96d4637) (source: https://en.wikipedia.org/wiki/Q-learning)

On every run the Q table is updated with a new Q value. This is defined in the above equation. This takes the existing value and multiplies it by the learned value. The learned value is a compibation of the reward from the latest move and the maximum Q value from the new state. In code this looks like

```python

maxFutureQ = np.max(qTable[newDiscreteState])  # estimate of optiomal future value
currentQ = qTable[discreteState + (action, )]  # old value

# formula to caculate all Q values
newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)

```

LEARNING_RATE and DISCOUNT are constants defined at the start of the script.
