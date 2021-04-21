# Reinforcement Learning Workshop: Q-Learning 🦾

Here you will learn how to develop and test Q-learning algorithms in the Open AI Gym.

## 🧰 Installation
You need **[Python](https://www.python.org/downloads/windows/) 3.5-3.8** (not 3.9). Also, you need to install these dependencies using the command prompt:
* (Check your python version: `python`)
* Numpy: `pip install numpy`
* Gym: `pip install gym`
* Matplotlib: `pip install matplotlib`

For this workshop we will be working with a local (classic) [Jupyter Notebook](https://jupyter.org/), which you can start by running `jupyter notebook` in the command prompt. It is possible to use gym in other text editors, however rendering results will work slightly different then.


## Step 1: Run the Environment (10 min.) 🏃🏽‍♀️

Take a look at the environments available in Gym [here](https://gym.openai.com/envs/#classic_control).
To get started, we will use the environment [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/).

**Start by creating a python file and `import` gym, numpy (`as np`), time, math and matplotlib.pyplot (`as plt`). Also, define `%matplotlib inline`. Then, you can use this basic example to render the environment:**
```python
env = gym.make('EnvironmentName')
env.reset()

for _ in range(Steps):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```
**Fill in the environment and the amount of steps you want to render.**
If you run the code you should see the cart move (window might open behind browser).

There might be a warning about calling step() that you can ignore for now. It means that you are rendering the agent even though your agent has already failed (or achieved) it's task and it is therefore `done`.


## Step 2: Explore the Environment Space (15 min.) 🛰️

Let's get to know our environment. Environments in gym comes with a predetermined `action_space` and `observation_space` that contain all valid actions and observations. If you build your own environment or use 3rd party environment, it might not have this data built in.

Use these functions to explore the CartPole-v1 space.
```python
print(env.action_space) # Number of actions
print(env.observation_space) # Array of n numbers
print(env.observation_space.high)
print(env.observation_space.low)
```
**Can you answer these questions with the functions above?:**

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

**Use `done` to solve the error from step 1 (fill in how many episodes you want to run):**
```python
for i_episode in range(Episodes):
    observation = env.reset()
    for t in range(Steps):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
```
**Is it better to finish an episode faster in this case?**


## Step 3: Variables in Q-Learning (10 min.) 🧮

The Q-table contains the expected reward for every possible action at any given state in the environment.

**Before you set up the Q-table you need to define some variables that will be important in the Q-learning process:**

**How high do you think the learning rate should be?**
```python
LEARNING_RATE = # 0-1
```
The learning rate defines how much you change your Q values after each step. In short: varying only a little from what you learned vs. taking big leaps.

**How high do you think the discount factor should be?**
```python
DISCOUNT = # 0-1 
```
The discount determines how much future events lose their value according to how far away in time they are. A discount factor of 0 means that you only care about immediate rewards. The higher your discount factor, the farther your rewards will propagate through time.

**How high do you think epsilon should be?**
```python
EPISODES = # How often you want to run the environment
total = 0 # reset value
total_reward = 0 # reset value
prior_reward = 0 # reset value

epsilon = # 0-1
epsilon_decay_value = 0.99995
```
Epsilon determines how much we are selecting actions based on past rewards vs. randomly. In short: exploitation vs. exploration.

To make it easier for our computer we also need to divide the observations (float numbers) into to a set amount of integer numbers (whole numbers).
```python
Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
```

## Step 4: Build a Q-Table and Start Training (15 min.) 📋🤓

**Set up the Q-table.**
```python
q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape
```

Define the function that will return the discrete observation we defined in the previous step.
```python
def get_discrete_state(state):
    discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))
```

**Next, write a for loop that goes through the episodes.**
```python
for episode in range(...):
    t0 = time.time() # Set the initial time
    discrete_state = get_discrete_state(env.reset()) # Get the discrete start for the restarted environment 
    done = False # Reset
    episode_reward = 0 # Reward starts as 0 for each episode

    if episode % 2000 == 0: 
        print("Episode: " + str(episode))
        
env.close()
```

**Tell the agent what to do and render every x episodes.**
```python
    while not done: 

        if np.random.random() > epsilon:

            action = np.argmax(q_table[discrete_state]) # Take coordinated action
        else:

            action = np.random.randint(0, env.action_space.n) # Take a random action

        new_state, reward, done, _ = env.step(action) # Step action to get new states, reward, and the "done" status.

        episode_reward += reward # Add the reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % x == 0:
            render()

        if not done: # update q-table
            max_future_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state
```

To improve training, prevent epsilon from decreasing if the current episode did worse than the one before it.
```python
    if epsilon > 0.05:
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))
```

Add a timer that counts the time the cart is able to balance the pole.
```python
    t1 = time.time() # Episode has finished
    episode_total = t1 - t0 # Episode total time
    total = total + episode_total

    total_reward += episode_reward # Episode total reward
    prior_reward = episode_reward

    if episode % SHOW_EVERY == 0: # Define SHOW_EVERY
        mean = total / SHOW_EVERY
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / SHOW_EVERY
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0
```

**How long does it take you to get to an average reward of 100?**


## Step 5: Plot Your Progress (15 min.) 📈

In order to decide if our RL agent has learned enough, you can visualize the average, minimum and maximum reward achieved during the last 1000 episodes.

**Use this (incomplete) example to plot the learning progress:**
```python
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
```
...
```python
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
        print()

env.close()

plt.figure(figsize=(9, 9))
# Plot here with matplotlib
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
```
**Why is it useful to know the maximum and minimum performance next to the average?**


## 🧪 Further Reading and Experimenting

### General
* garychl. [Applications of Reinforcement Learning in Real World](https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12)
* sentdex. [Creating A Reinforcement Learning (RL) Environment - Reinforcement Learning p.4](https://www.youtube.com/watch?v=G92TF4xYQcU&t=2857s)

### Deep RL
* M. Wang. [Deep Q-Learning Tutorial: minDQN](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)
* OpenAI. [Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)

### Model-based RL
D. Ha, J. Schmidhuber. [World Models](https://worldmodels.github.io/)
