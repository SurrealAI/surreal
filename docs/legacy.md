# Steps to add a new RL algorithm

Read the inline docstrings of each class in the source code for more information:

## Step 1: Model

Create a model that extends `utils.pytorch.Module`, which is an extension of the PyTorch Module. This model represents the policy network or Q function.

## Step 2: Agent

Extend the `agent.base.Agent` class.

Required override:
- `_act()`

Optional override:
- `initialize()`
- `close()`

## Step 3: Learner

Extend the `learner.base.Learner` class.

Required override:
- `learn()`

## Step 4: Replay

Extend the `replay.base.Replay` class.

Required override:
- `_insert()`
- `_sample()`
- `start_sample_condition()`
- `aggregate_batch()`

Optional override:
- `_evict()`


# Installation

## Main surreal library

```
cd Surreal/
pip install -e surreal
```

## Redis

```
brew install redis

$ redis-server  # runs on localhost with default port
```

## Docker and Kubernetes

TODO

## Run Cartpole

```
# make sure redis-server is running locally at the default port
# must run the following two commands in order
python surreal/main/run_cartpole_learner.py
python surreal/main/run_cartpole_agent.py
```

# Detailed API

**will be moved to another file**

## Agent-side components

Agent-side code listens for updates from the parameter server and sends experiences to the distributed Replay pool. 

### Model 

A neural network that extends the `model.base.Model` class.  

The model is responsible for forward/backward prop and parameter serialization/deserialization, but not the action selection. 

### Agent

An agent wraps the Model class and selects actions that will be executed in the environment. The actions are not necessarily the same as the `Model` output, because the policy can be epsilon-greedy, for example. 

Agent also encapsulates exploration logic. It needs to have support for both the training and evaluation mode, e.g. epsilon greedy during training but `argmax` at evaluation. 
 
### ExpSender
 
 Send experience tuples to the distributed Replay pool, i.e. a Redis server (or cluster of servers) that holds all the observation in memory. 
 
### TorchListener
 
 Listens for new updates from the parameter server, and update the `Model` inside the `agent` accordingly. 
 
 Warning: make sure you lock the agent so that "updating parameters" and "forward prop" do not interleave. Use `threading.Lock()`
 
### Environment
 
Extends `envs.base.Env` class. Wrappers extend `envs.base.Wrapper` class.
 
Key difference from OpenAI Gym API 

 -  `reset()` function now returns a tuple `(obs, info)` instead of just `obs`.
 - Support self-play and multi-agent (future). 
 - We rely heavily on the catch-all `info` dict to make env and agents as versatile as possible. In Gym, `info` contains nothing but unimportant diagnostics, and is typically empty altogether. In contrast, we will put crucial information such as the individual frames in `info` when doing frame-stacking, because we don't want to duplicate each frame many times in the Redis storage. The other scenario is multi-agent training. The `info` dict will likely have rich contents. 
 
 
## Learner-side components

### One-step learner 

Learners extend `learners.base.Learner` class. It performs one batch of policy optimization at each `.optimize(...)` call. 

### Replay

Replay pools extend `replay.base.Replay` class. It keeps an internal data structure of the replay buffer, which can be as complicated as max-sum-trees that execute insertion at O(log(n)). 

Under the hood, replay objects are also responsible for communicating with the Redis replay server asynchronously. The end user does not need to know anything about the threads, which will start running in background when they call `replay.start_threads()`. More specifically, the threads obey the "producer-consumer pattern": 

1. `PointerQueue` enqueue thread that downloads the experience tuples with observation pointers from the replay server.

2. `PointerQueue` dequeue thread that inserts the experience tuples into the local replay object. The insertion logic can be very complicated, like the max-sum-tree data structure in Prioritized Experience Replay.

3. `ExpFetcherQueue` enqueue thread that downloads the actual observations from their pointers. This thread also aggregates the batch of downloaded observations into PyTorch-ready tensors. The learner will pull from this queue to retrieve the next mini-batch.

Note that `ExpFetcherQueue`'s dequeue method is _synchronous_, because it is invoked by the learner every time it needs the next mini-batch. If the rate of agents producing experience is slower than the rate of learners consuming experience (i.e. backprop), the call to `dequeue` will wait until the next `batch_size` of samples arrive. The `dequeue` method is wrapped in `replay.batch_iterator()`.

In the release, we will provide at least the following instantiations of the abstract `Replay` class:

- UniformReplay: vanilla DQN. 
- PrioritizedReplay: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).
- MixedReplay: [Interpolated Policy Gradient](https://arxiv.org/pdf/1706.00387.pdf).

### TorchBroadcaster

Broadcasts the latest parameters to the parameter server.
Called inside the learner optimization loop.

## Main entry points

### Workflow: `run_agent.py`

#### Purpose

The `run_agent` script does NOT do any learning. It is only responsible for generating experiences.

#### Initialization

1. Initialize the environment instance with monitoring and visualization wrappers. 

2. Initialize a model from scratch or saved checkpoint. 

3. Create an agent with the model we just created.

4. Specify the exploration strategy for the agent. Because each agent is independent of each other in the cluster, they can adopt different exploration strategies to navigate the state space more efficiently. 

5. Create an instance of `RedisClient` that connects to the replay server address. 

6. Create an `ExpSender` object that wraps around the `RedisClient` above. 
 
7. Create another instance of `RedisClient` that connects to the parameter server address. 

8. Create a `TorchListener` from the `RedisClient` above and the `model` object. The listener will continuously listen for broadcasted messages from the parameter server and push the new parameters into the `model`. 

#### Main loop

Now we are ready to run the agent-environment interaction loop.  

1. Start the listener thread in the background. `listener.run_listener_thread()`. 

2. Receives initial observation and info by calling `env.rest()`. 

3. For each iteration, the `agent` calls `.act(obs)` and returns the selected action (can be random according to the exploration strategy). 

4. Call `env.step(action) -> obs, reward, done, info`. 

5. Call `exp_sender.send([obs], action, reward, done, info)` to upload the experience tuple to the replay server.

6. Do book-keeping, like logging the current reward streaks, speed `iter/s`, number of steps, etc.
 
 
#### Skeleton code

```python
env = Visualization(EpisodeMonitor(Atari('SpaceInvaders')))
q_model = QFunction(hiddens=[64, 32], action_space=6)
exploration = LinearSchedule(...)
q_agent = QAgent(q_model, exploration)
q_agent.set_mode('training')

replay_client = RedisClient('127.0.0.1:6060') # replay server IP
exp_sender = ExpSender(replay_client)

ps_client = RedisClient('192.168.6.6:8080') # parameter server IP
listener = TorchListener(ps_client, q_model)
listener.run_listener_thread()

obs, info = env.reset()
while True:
    action = q_agent.act(obs)
    new_obs, reward, done, info = env.step(action)
    exp_sender.send([obs, new_obs], action, reward, done, info)
    obs = new_obs
```

### Workflow: `run_learner.py`

#### Purpose

The learner script updates the parameters by sampling from the replay pool. It is unaware of the env or agent exploration. It broadcasts the latest parameters after every minibatch. 

#### Initialization

1. Initialize a model from scratch or saved checkpoint. 

2. Create an instance of `RedisClient` that connects to the replay server address. 

3. Create an object that inherits the `replay.base.Replay` class. Initialize with the above client and other settings of the replay, such as batch size, start sampling condition, prioritization hyper-parameters, etc. 
 
4. Create another instance of `RedisClient` that connects to the parameter server address. 

5. Create a `TorchBroadcaster` instance with the above client. 

6. Initialize the learner with the `model` and `replay` objects. 

#### Main loop

1. `replay.start_threads()` to start various background jobs together.
    1. `PointerQueue` enqueue thread. 
    2. `PointerQueue` dequeue thread.  
    3. `ExpFetcherQueue` enqueue thread. 
    
2. For-loop over `replay.batch_iterator()`. Each iteration returns the next minibatch as PyTorch-ready tensors. 

3. `learner.optimize(batch)`

4. At every few steps, call `broadcaster.broadcast(model)` to push the latest params to the parameter server. 

#### Skeleton code

```python
# model spec must be the same as in agents
q_model = QFunction(hiddens=[64, 32], action_space=6)

replay_client = RedisClient('127.0.0.1:6060') # replay server IP
replay = UniformReplay(replay_client, batch_size=64)

ps_client = RedisClient('192.168.6.6:8080') # parameter server IP
broadcaster = TorchBroadcaster(ps_client, q_model)

dqn = DQNLearner(q_model, replay)

replay.start_threads()
for i, batch in replay.batch_iterator():
    dqn.optimize(i, batch)
    if i % BROADCAST_FREQUENCY == 0:
        broadcaster.broadcast(q_model, tag_message)
```

### Cluster management

- Scripts to start Redis server.
- The above can all be wrapped in Kubernetes (@Zihua)


## Other components

### Visualization panel

- [Visdom](https://github.com/facebookresearch/visdom). 
- [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) for PyTorch. 

### Evaluation process

Environment and agents can behave differently in training mode than in evaluation mode. On the env side, Atari games have many wrappers (such as cap to only 1 life) that make them easier to train. Those wrappers will be removed during eval. 

On the agent side, the agent will do `argmax` or other different behaviors for eval instead of epsilon-greedy policy.   

Evaluation can run infrequently on a separate process than the agents and learners.
It will also take care of book-keeping:

 - Evaluation score tracking
 - Checkpointing
 - Logging
 - Visualization board

#### Skeleton code

```python
env = Visualization(EpisodeMonitor(Atari('SpaceInvaders')))
q_model = QFunction(hiddens=[64, 32], action_space=6)
q_agent = QAgent(q_model, exploration=None)
q_agent.set_mode('eval')

ps_client = RedisClient('192.168.6.6:8080') # parameter server IP
listener = TorchListener(ps_client, q_model)
listener.run_listener_thread()

obs, info = env.reset()
while True:
    action = q_agent.act(obs)
    new_obs, reward, done, info = env.step(action)
    obs = new_obs
    # book-keeping code goes here
```

### Distributed logging


