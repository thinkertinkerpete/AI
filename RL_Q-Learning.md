# Reinforcement Learning Workshop: Q-Learning 🦾

Here you will learn how to develop and test Q-learning algorithms in the Open AI Gym.

## Installation
You need **[Python](https://www.python.org/downloads/windows/) 3.5-3.8** (not 3.9). Also, you need to install these dependencies using the command prompt:
* (Check your python version: `python`)
* Numpy: `pip install numpy`
* Gym: `pip install gym`
* Matplotlib: `pip install matplotlib`

For this workshop we will be working with a local (classic) [Jupyter Notebook](https://jupyter.org/), which you can start by running `jupyter notebook` in the command prompt. It is possible to use gym in other text editors, however rendering results will work slightly different then.


## Let's Get Started!

### Run an Environment (5 min.) 🏃🏽‍♀️
Take a look at the environments available in Gym [here](https://gym.openai.com/envs/#classic_control).
To get started, we will use the environment [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/).

Start by creating a python file and import gym, numpy (`as np`) and matplotlib.pyplot (`as plt`). Also, define `%matplotlib inline`. Then, you can use this minimal example to render the environment:
```python
env = gym.make('EnvironmentName')
env.reset()

for _ in range(Steps):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```
Fill in the environment and the amount of steps you want to render.
If you run the code you should see the cart move.

There might be a warning about calling step() that you can ignore for now. It means that you are rendering the agent even though your agent has already failed (or achieved) it's task and it is therefore `done`.

### Explore the Environment Space (10 min.) 🛰️
Let's get to know our environment. Environments in gym comes with a predetermined `action_space` and `observation_space` that contain all valid actions and observations. If you build your own environment or use 3rd party environment, it might not have this data built in.

For this workshop, we want to know:
* How many actions can the agent take?
* ...
* What is the highest possible observation?
* What is the lowest possible observation?

To answer these questions use the functions below.
```python
print(env.action_space) # Number of actions
print(env.observation_space) # Array of n numbers
print(env.observation_space.high)
print(env.observation_space.low)
```
