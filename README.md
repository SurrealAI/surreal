# Design

Read the source code doc strings of each class for more information.

## Configs

The major components, such as `Agent`, `Learner`, and `Replay`, are all configured by the `surreal.session.Config` objects.

### Semantics

`Config` is essentially dict with attribute access. For example, `Config.myvalue` is equivalent to `Config['myvalue']`, and you can perform the same dict operations on Config, e.g. `.keys()`, `.items()`, etc. `Config` behaves very much like [EasyDict](https://github.com/makinacorpus/easydict).

Besides the usual dict methods, `Config` has the following extra methods:

- `load_file` (class method): loads from JSON or YAML according to the file extension.
- `dump_file`
- `extend`: extends a default "base config". Think of it as a JSON equivalent of "python default keyword arguments".

Example of default config:
```python
default_config = Config({
    'redis': {
        'replay': {
            'host': 'localhost',
            'port': 6379,
        },
        'ps': {
            'host': '192.168.11.11',
            'port': 8888,
        },
    },
    'obs_spec': {
        'type': 'continuous',
        'dim': [64, 64, 3]
    }
})

my_config = Config({
    'redis': {
        'replay': {
            'host': 'myreplayhost'
        },
        'ps': {
            'port': 12345
        }
    },
    'obs_spec': {
        'extra': 'special_extra'
    }
})

my_config.extend(default_config)

# Now my_config will be filled by the defaults:
assert my_config == Config({
    'redis': {
        'replay': {
            'host': 'myreplayhost',
            'port': 6379,
        },
        'ps': {
            'host': '192.168.11.11',
            'port': 12345
        }
    },
    'obs_spec': {
        'type': 'continuous',
        'dim': [64, 64, 3],
        'extra': 'special_extra'
    }
})
```

The base config can specify special placeholders. `extend()` will check if the input config satisfies the placeholder specification in the default config. Any entry marked by the placeholder is required and will raise `ConfigError` if it's missing from the input config.

The following placeholders are supported:

- `_object_`: matches anything.
- `_singleton_`: matches any object that is not a list or dict.
- `_dict_`
- `_list_`
- `_int_`
- `_float_`
- `_num_`: matches both `int` and `float`
- `_bool_`
- `_str_`
- `_enum[option1, option2, ...]_`: matches a string if it is one of the options. The enum options are enclosed in square brackets and separated by comma.


Example of placeholder semantics:
```python
default_config = Config({
    'redis': {
        'replay': {
            'host': 'localhost',
            'port': 6379,
            'flag': '_bool_'
        },
        'ps': {
            'host': '_str_',
            'port': '_int_',
        },
    },
    'obs_spec': {
        'type': '_enum[continuous,discrete]_',
        'dim': '_list_'
    }
})

my_config = Config({
    'redis': {
        'replay': {
            'flag': False
        },
        'ps': {
            'host': 'mypshost',
            'port': 12345
        }
    },
    'obs_spec': {
        'type': 'discrete',
        'dim': [128, 128]
    }
})

my_config.extend(default_config)

assert my_config == Config({
    'redis': {
        'replay': {
            'host': 'localhost',
            'port': 6379,
            'flag': False
        },
        'ps': {
            'host': 'mypshost',
            'port': 12345
        }
    },
    'obs_spec': {
        'type': 'discrete',
        'dim': [128, 128]
    }
})
```

### Surreal configs

All Surreal experiments boil down to 3 configs. The default base configs can be found in `session/default_configs.py`.

**You can find concrete examples in `main.cartpole_configs`.**

Let's take DQN as example:

#### 1. learn_config

Anything related to agent, model, training hyperparameters, and replay buffer. Typically have the following sub-dict:

- `model`: architecture of the Q function. Depth of conv layers, filter widths, etc.
- `algo`: hyperparameters in Q learning. Learning rate, discounting factor, Q-target update frequency, etc.
- `replay`: replay memory settings. Memory size, priority schedule, eviction policy, etc. 
- `sender`: agent side calls `env.ExpSenderWrapper` to send experience tuples to the replay server. Sender config includes obs cache size, max redis replay queue size, etc. 

#### 2. env_config

- `action_spec`: provides information to the model builder. For Gym envs, `action_spec` is redundant, but must be consistent with their `action_space`. 
    - `type`: `continuous` or `discrete`. 
    - `dim`: a list that specifies the action shape in continuous control, or a singleton list of the action dimension in discrete tasks. 
   
- `obs_spec`: provides information to the model builder. For Gym envs, `obs_spec` is redundant, but must be consistent with their `obs_space`. 
    - `type`: TODO
    
    
#### 3. session_config

Configures the cluster, monitor, evaluator, and logger. Takes care of all book-keeping. 

- `redis`: IP addresses of the replay server(s) and parameter server(s).
- `logger`: distributed logging (TODO)
- `monitor`: video monitoring (TODO), tensorboard visualization
- `evaluator`: tracks evaluation performance, checkpoints the best parameters.

Please refer to `session.default_configs.BASE_SESSION_CONFIG`.


## Agent

To add a new agent, please extend `agent.base.Agent` class. Example `agent/q_agent.py`.

Make sure all the NNs in the agent inherits from `utils.pytorch.Module`, **NOT** the native `torch.nn.Module`!!

Override the following methods:

- `act(obs)`: returns an action upon seeing the observation.
- `module_dict()`: returns a dict of `name -> utils.pytorch.Module`. The dict must be consistent with `learner.module_dict()` for the parameter server to work correctly. 
- `default_config()`: specify the agent's defaults for `learn_config`. 
- `close()`: clean up

Public entry API:

- `act(obs)`
- `close()`

Agent also encapsulates exploration logic. It needs to have support for both the training and evaluation mode, e.g. epsilon greedy during training but `argmax` at evaluation. 
Agent process should use `AgentMode.training` while evaluator process should use `AgentMode.eval_***`. 
All enums must inherit from `utils.StringEnum`.

```python
class AgentMode(StringEnum):
    training = ()
    eval_stochastic = ()
    eval_deterministic = ()
```


## Learner

To add a new learner, please extend `learner.base.Learner` class. Example `learner/dqn.py`. 

Override the following methods:

- `learn(batch_exp)`: takes a batch of experience and performs one iteration of the optimization loop. The input is typically an `EasyDict` with the following batched values:
    - obs
    - obs_next
    - actions
    - rewards
    - dones
- `module_dict()`: returns a dict of `name -> utils.pytorch.Module`. Because the values are broadcasted to Redis, the `module_dict` must be consistent with `agent.module_dict()` for the parameter server to work correctly. 
- `default_config()`: specify the learner's defaults for `learn_config`. 
- `save(file_path)`: saves the learned parameters to `file_path`. TODO: save() should be triggered by a remote notification from the evaluator, because the learner process doesn't do book-keeping. 

Public entry API:

- `learn(batch_exp)`
- `broadcast(message='')`: pushes the latest parameters to the parameter server.
- `save(file_path)`

## Replay

To add a new replay, please extend `replay.base.Replay` class. Example `replay/uniform_replay.py`. 
Replay object keeps an internal data structure of the replay buffer, which can be as complicated as max-sum-trees that execute insertion at O(log(n)). 


Override the following methods:

- `_insert(exp_dict)`: add a new experience to the replay. Implements _passive eviction_ logic if memory capacity is exceeded.
- `_sample(batch_size)`: samples from internal memory data structure and returns a list of `exp_dict`. 
- `_evict(*args, **kwargs)`: _active eviction_. Returns a list of `exp_dict` to be evicted.
- `_start_sample_condition()`: returns True if we are ready to start sampling, e.g. when the replay memory has accumulated more than 1000 `exp_dict`. 
- `_aggregate_batch(list_of_exp_dict)`: aggregate a batch into PyTorch tensors for the learner. Returns `batch_exp`, an `EasyDict` of entries consistent with `learner.learn(batch_exp)` (see [Learner API](#learner)).


Public entry API:

- `sample()`: returns `batch_exp`. Note that `batch_size` is specified in `learn_config` at initialization, so we don't pass it again as an arg.
- `sample_iterator()`: infinite iterator that wraps around `sample()`.
- `evict(*args, **kwargs)`: active eviction. 
- `insert(exp_dict)`: not typically called by hand. `insert()` runs in the background. 

All the background threads are automatically started at initialization. 

TODO: add more explanation about the background threads. Meanwhile you can read about them in [legacy.md](docs/legacy.md).


## Environment
 
To add a new environment, please extend `env.base.Env` class. 

Wrappers extend `env.base.Wrapper` class.
 
Key difference from OpenAI Gym API 

 -  `reset()` function now returns a tuple `(obs, info)` instead of just `obs`.
 - `metadata` is a class-level dict that passes on to derived classes.
 - `obs_spec` and `action_spec` are now passed from the env config instead of being instance attributes. There will be no `action_space` and `obs_space` attributes as in Gym. 
 - We rely heavily on the catch-all `info` dict to make env and agents as versatile as possible. In Gym, `info` contains nothing but unimportant diagnostics, and is typically empty altogether. In contrast, we will put crucial information such as the individual frames in `info` when doing frame-stacking, because we don't want to duplicate each frame many times in the Redis storage. The other scenario is multi-agent training. The `info` dict will likely have rich contents. 
 
#### env.GymAdapter

Wraps any Gym env into a Surreal-compatible env. Note that you still have to make sure the `obs_spec` and `action_spec` in the env config are consistent with the Gym env. The code does not enforce the constraint. Your model builder will build the wrong NN if the specs are wrong. 

#### env.ExpSenderWrapper

Encapsulates the `ExpSender` that sends experience dicts to the replay server. Each `env.step()` call will connect to the network. 

The wrapper takes `learn_config` and `session_config`. Make sure the `learn_config` dict includes a section of `"sender"`: 

```python
{
    'sender': {
        'pointers_only': True,
        'save_exp_on_redis': False,
        'max_redis_queue_size': 10000,
        'obs_cache_size': 10000,
    },
    # ... other configs ...
}
```

#### env.Monitor

TODO: the current `env.EpisodeMonitor` is very crude. It doesn't do any visualization.


## Session

Session is not a singleton, but a collection of components that take care of all kinds of book-keeping and cluster management. 
  
### Evaluator

TODO

Environment and agents can behave differently in training mode than in evaluation mode. On the env side, Atari games have many wrappers (such as cap to only 1 life) that make them easier to train. Those wrappers will be removed during eval. 

On the agent side, the agent will do `argmax` or other different behaviors for eval instead of epsilon-greedy policy.   

Evaluation can run infrequently on a separate process than the agents and learners.
It will also take care of book-keeping:

 - Evaluation score tracking
 - Checkpointing
 
 
### Tensorboard
 
TODO

### Kubernetes
 
TODO



# Workflow

## A Tale of Three Configs

```python
cartpole_learn_config = {
    'model': {
        'convs': [],
        'fc_hidden_sizes': [128],
        'dueling': False
    },
    'algo': {
        'lr': 1e-3,
        'optimizer': 'Adam',
        'grad_norm_clipping': 10,
        'gamma': .99,
        'target_network_update_freq': 250 * 64,
        'double_q': True,
        'exploration': {
            'schedule': 'linear',
            'steps': 30000,
            'final_eps': 0.01,
        },
        'prioritized': {
            'enabled': False,
            'alpha': 0.6,
            'beta0': 0.4,
            'beta_anneal_iters': None,
            'eps': 1e-6
        },
    },
    'replay': {
        'batch_size': 64,
        'memory_size': 100000,
        'sampling_start_size': 1000,
    },
    'sender': {
        'pointers_only': True,
        'save_exp_on_redis': False,
        'max_redis_queue_size': 100000,
        'obs_cache_size': 100000,
    }
}


cartpole_env_config = {
    'action_spec': {
        'dim': [2],
        'type': 'discrete'
    },
    'obs_spec': {
        'dim': [4],
    }
}


cartpole_session_config = {
    'redis': {
        'replay': {
            'name': 'replay',
            'host': 'localhost',
            'port': 6379,
        },
        'ps': {
            'name': 'ps',
            'host': '192.168.0.0',
            'port': 8888,
        },
    },
}
```

## Agent side main script

1. Create the env from `env_config`.
2. Wrap it with `env.ExpSenderWrapper`. Redis replay server must be up and running by now. `learn_config` should include a `"sender"` section. 
3. Create the agent from all 3 configs and set to `AgentMode.training`.
4. Start env loop. 

```python
env = GymAdapter(gym.make('CartPole-v0'))
env = ExpSenderWrapper(
    env,
    learn_config=cartpole_learn_config,
    session_config=cartpole_session_config
)
env = EpisodeMonitor(env)

q_agent = QAgent(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=cartpole_session_config,
    agent_mode=AgentMode.training,
)
obs, info = env.reset()
for T in itertools.count():
    action = q_agent.act(U.to_float_tensor(obs))
    obs, reward, done, info = env.step(action)  # sends to Redis automatically
    if done:
        obs, info = env.reset()
```

## Learner side main script

1. Create the replay data structure from all 3 configs. `learn_config` should include a `"replay"` section.
2. Create the learner from all 3 configs.
3. `replay.sample_iterator()` loop. 
4. Call `learner.broadcast()` periodically to push the latest parameters to Redis server. 


```python
replay = UniformReplay(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=cartpole_session_config
)
dqn = DQNLearner(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=cartpole_session_config
)
for i, batch in enumerate(replay.sample_iterator()):
    td_error = dqn.learn(batch)  # doesn't have to return td_error
    if (i+1) % 100 == 0:
        dqn.broadcast(message='hello surreal '+str(i))
```


# Installation

## Main surreal library

```
cd Surreal/
pip install -e .
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
