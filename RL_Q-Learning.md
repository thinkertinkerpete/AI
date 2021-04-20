# Reinforcement Learning Workshop: Q-Learning 🦾

Here you will learn how to develop and test Q-learning algorithms in the Open AI Gym.

## Installation
You need **[Python](https://www.python.org/downloads/windows/) 3.5-3.8** (not 3.9). Also, you need to install these dependencies using the command prompt:
* (Check your python version: `python`)
* Numpy: `pip install numpy`
* Gym: `pip install gym`
* Matplotlib: `pip install matplotlib`

For this workshop we will be working with a local (classic) [Jupyter Notebook](https://jupyter.org/), which you can start by running `jupyter notebook` in the command prompt. It is possible to use gym in other text editors, however rendering results will work slightly different then.


## Step 1: Run an Environment (5 min.) 🏃🏽‍♀️
Take a look at the environments available in Gym [here](https://gym.openai.com/envs/#classic_control).
To get started, we will use the environment [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/).

**Start by creating a python file and import gym, numpy (`as np`) and matplotlib.pyplot (`as plt`). Also, define `%matplotlib inline`.**
Then, you can use this basic example to render the environment:
```python
env = gym.make('EnvironmentName')
env.reset()

for _ in range(Steps):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```
**Fill in the environment and the amount of steps you want to render.**
If you run the code you should see the cart move.

There might be a warning about calling step() that you can ignore for now. It means that you are rendering the agent even though your agent has already failed (or achieved) it's task and it is therefore `done`.

## Step 2: Explore the Environment Space (10 min.) 🛰️
Let's get to know our environment. Environments in gym comes with a predetermined `action_space` and `observation_space` that contain all valid actions and observations. If you build your own environment or use 3rd party environment, it might not have this data built in.

Use these functions to explore the CartPole-v1 space.
```python
print(env.action_space) # Number of actions
print(env.observation_space) # Array of n numbers
print(env.observation_space.high)
print(env.observation_space.low)
```
Can you answer these questions with the functions above?:

1. How many actions can the agent take?
1. What are these actions?
1. How many observations are there?
1. What do the observations measure?
1. What is the highest possible observation?
1. What is the lowest possible observation?

The `step` function contains even more information. It contains these four values (Ref.: [Open AI](https://gym.openai.com/docs/)):

> * **`observation` (object)**: an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
> * **`reward` (float)**: amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
> * **`done` (boolean)**: whether it’s time to `reset` the environment again. Most (but not all) tasks are divided up into well-defined episodes, and `done` being `True` indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
> * **`info` (dict)**: diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change).

Use `done` to solve the error from step 1:

