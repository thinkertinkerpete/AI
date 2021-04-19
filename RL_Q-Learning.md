# Reinforcement Learning Workshop: Q-Learning 🦾

Here you will learn how to develop and test Q-learning algorithms in the Open AI Gym.

## Installation
You need these dependencies:
* Numpy: `pip install numpy`
* Gym: `pip install gym`
* Matplotlib: `pip install matplotlib`
* [Python](https://www.python.org/downloads/windows/) 3.5-3.8: `python`

If you work in Google Colab you need to put a `!` in front of these command prompt commands.


## Let's Get Started!

### Training Environments 🏋🏿
Take a look at the environments available in Gym [here](https://gym.openai.com/envs/#classic_control).

To get started, we will use the environment [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/).

### Run an Environment 🏃🏽‍♀️
Start by creating a python file and import gym, numpy and matplotlib. You can use this minimal example to render the environment:
```python
env = gym.make('EnvironmentName')
env.reset()

for _ in range(Steps):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```
Fill in the environment and the amount of steps you want to render.
If you run the code you should see the agent (car/pendulum/cart) move.

There might be a warning about calling step(). It means that you are rendering the agent even though your agent has already failed (or achieved) it's task and it is therefore `done`.
We can solve this by stating `done = False` in the beginning and adding a `while not done:` loop in the for loop.
