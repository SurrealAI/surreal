# Design

This is a quick overview of Surreal core v0.2. 

For a fully working example, please look at `main/dqn_cartpole/`, which uses the main functions in `main/basic_boilerplate.py` to instantiate the agent, learner, and replay.

The configs are in `main/dqn_cartpole/configs.py`, which extends the default configs at `session/default_configs.py`. 

You can launch and manage the entire system with `main/cluster_dashboard.ipynb`. You must have `tmux` installed. If you use virtualenv, please uncomment the parameter `preamble_cmd='source activate myenv'` to `TmuxCluster`.

## Agent

To add a new agent, please extend `agent.base.Agent` class. Example `agent/q_agent.py`.

Make sure all the NNs in the agent inherits from `utils.pytorch.Module`, **NOT** the native `torch.nn.Module`!!

Override the following methods:

- `act(obs)`: returns an action upon seeing the observation.
- `module_dict()`: returns a dict of `name -> utils.pytorch.Module`. The dict must be consistent with `learner.module_dict()` for the parameter server to work correctly. 
- `default_config()`: specify the agent's defaults for `learner_config`.

Public methods:

- `act(obs)`
- `fetch_parameter()`: pull latest parameters from the parameter server.
- `fetch_parameter_info()`: pull the latest parameter info, returns a dict with
    - `time`: time stamp of the parameter, i.e. `time.time()`.
    - `iteration`: provided by the learner.
    - `message`: optional message sent by the learner. Empty string is default.
    - `hash`: hash signature of the parameter binary blob.
    
- `update_tensorplex(tag_value_dict)`: update distributed Tensorboard every interval. The interval is specified in `session_config`.

Public attribute:

- `.log`: [Loggerplex](https://github.com/stanfordvl/tensorplex) instance. You can record any agent activity with it by calling `self.log.info("info message")`, `self.log.warning("something bad happened")`, etc.
- `.tensorplex`: instead of interacting with this attribute, you typically call `update_tensorplex()` method, which periodically sends the info to the Tensorplex server. You can directly call `self.tensorplex.add_scalar(...)` if you want more control.

Agent also encapsulates exploration logic. It needs to have support for both the training and evaluation mode, e.g. epsilon greedy during training but `argmax` at evaluation. 
Agent process should use `AgentMode.training` while evaluator process should use `AgentMode.eval_***`. 
All enums must inherit from `utils.StringEnum`.

```python
class AgentMode(StringEnum):
    training = ()
    eval_stochastic = ()
    eval_deterministic = ()
```

Agent main entry sample code:

```python
env = ExpSenderWrapper(
    env,
    session_config=session_config
)
env = TrainingTensorplexMonitor(
    env,
    agent_id=agent_id,
    session_config=session_config,
    separate_plots=True
)

agent = MyAgent(
    learner_config=learner_config,
    env_config=env_config,
    session_config=session_config,
    agent_id=agent_id,
    agent_mode=agent_mode,
)

obs, info = env.reset()
while True:
    action = agent.act(U.to_float_tensor(obs))
    obs, reward, done, info = env.step(action)
    if done:
        obs, info = env.reset()
        agent.fetch_parameter()

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
- `module_dict()`: returns a dict of `name -> utils.pytorch.Module`. Because the values are broadcasted (PUBSUB pattern) to the parameter server node, the `module_dict` must be consistent with `agent.module_dict()` for the parameter server to work correctly. 
- `default_config()`: specify the learner's defaults for `learner_config`.
- `save(file_path)`: saves the learned parameters to `file_path`. TODO: save() should be triggered by a remote notification from the evaluator, because the learner process doesn't do book-keeping. 

Public methods:

- `learn(batch_exp)`
- `publish_parameter(iteration, message='')`: pushes the latest parameters to the parameter server. Message is optional.
- `save(file_path)`
-  `update_tensorplex(tag_value_dict)`: same as agent's.

Public attribute:

- `.log`: [Loggerplex](https://github.com/stanfordvl/tensorplex) instance. You can record any learner activity with it by calling `self.log.info("info message")`, `self.log.warning("something bad happened")`, etc.
- `.tensorplex`: instead of interacting with this attribute, you typically call `update_tensorplex()` method, which periodically sends the info to the Tensorplex server. You can directly call `self.tensorplex.add_scalar(...)` if you want more control.

Learner main entry:

```python
learner = MyLearner(
    learner_config=learner_config,
    env_config=env_config,
    session_config=session_config,
)

for i, batch in enumerate(learner.fetch_iterator()):
        learner.learn(batch)
        learner.publish_parameter(i, message='batch '+str(i))
```

## Replay

To add a new replay, please extend `replay.base.Replay` class. Example `replay/uniform_replay.py`. 
Replay object keeps an internal data structure of the replay buffer, which can be as complicated as max-sum-trees that execute insertion at O(log(n)). 


Override the following methods:

- `insert(exp_tuple)`: add a new experience to the replay. `exp_tuple` is defined as `namedtuple('ExpTuple', 'obs action reward done info')`. Please implement _passive eviction_ logic if memory capacity is exceeded. Simply remove the evicted tuple from the internal memory. v0.2 doesn't have Redis so we don't need to manually ensure consistency with an external database.
- `sample(batch_size)`: samples from internal memory data structure and returns a list of `ExpTuple`. 
- `evict()`: _active eviction_. Implement logic to remove `ExpTuple` from internal memory. A periodic eviction can be scheduled in the background.
- `start_sample_condition()`: returns True if we are ready to start sampling, e.g. when the replay memory has accumulated more than 1000 experiences. 
- `aggregate_batch(list_of_exp_tuples)`: aggregate a batch into PyTorch tensors for the learner. Returns `batch_exp`, an `EasyDict` of entries consistent with `learner.learn(batch_exp)` (see [Learner API](#learner)).

Public attribute:


Public attribute:

- `.log`: [Loggerplex](https://github.com/stanfordvl/tensorplex) instance. You can record any replay activity with it by calling `self.log.info("info message")`, `self.log.warning("something bad happened")`, etc.
- `.tensorplex`: call `self.tensorplex.add_scalar(...)` to display any stats you care about.

TODO: the eviction thread needs to be configured in Config.

Replay main entry:

```python
replay = MyReplay(
    learner_config=learner_config,
    env_config=env_config,
    session_config=session_config,
)
replay.start_threads()
```

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

Encapsulates the `ExpSender` that sends experience dicts to the replay server. Each `env.step()` call will talk to the network. 

The wrapper takes `learner_config` and `session_config`. Make sure the `session_config` dict includes a section of `"sender"`. To minimize network latency, the sender buffers a number of experience tuples before it sends them together in a larger chunk. The buffer size is `flush_iteration`.

```python
{  # session_config
    'sender': {
        'flush_iteration': 100,
        'flush_time': # TODO not implemented
    },
}
```

#### env.TensorplexMonitor

- `env.TrainingTensorplexMonitor`
- `env.EvalTensorplexMonitor`

sends information like reward and step/s to Tensorplex.

#### env.ConsoleMonitor

Summarizes the same information as `TensorplexWrapper`, but prints to the console as a neat table instead of sending over network. Useful for local debugging without Tensorplex server.

## Session

Session is not a singleton, but a collection of components that take care of all kinds of book-keeping and cluster management. 
  
### Evaluator

Environment and agents can behave differently in training mode than in evaluation mode. On the env side, Atari games have many wrappers (such as cap to only 1 life) that make them easier to train. Those wrappers will be removed during eval. 

On the agent side, the agent will do `argmax` or other different behaviors for eval instead of epsilon-greedy policy.   

Evaluation can run infrequently on a separate process than the agents and learners.
 
 ### Cluster dashboard

`main/cluster_dashboard.ipynb`

# Config system

The major components, such as `Agent`, `Learner`, and `Replay`, are all configured by the `surreal.session.Config` objects.

## Semantics

`Config` is essentially dict with attribute access. For example, `Config.myvalue` is equivalent to `Config['myvalue']`, and you can perform the same dict operations on Config, e.g. `.keys()`, `.items()`, etc. `Config` behaves very much like [EasyDict](https://github.com/makinacorpus/easydict).

Besides the usual dict methods, `Config` has the following extra methods:

- `load_file` (class method): loads from JSON or YAML according to the file extension.
- `dump_file`
- `extend`: extends a default "base config". Think of it as a JSON equivalent of "python default keyword arguments".

Example of default config:
```python
default_config = Config({
    'demo': {
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
    'demo': {
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
    'demo': {
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
    'demo': {
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
    'demo': {
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
    'demo': {
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

## Surreal config options

All Surreal experiments boil down to 3 configs. The default base configs can be found in `session/default_configs.py`.

**A quick example is in `main/dqn_cartpole/config.py`.**

Let's start with DQN:

### 1. learner_config

Anything related to agent, model, training hyperparameters, and replay buffer. Typically have the following sub-dict:

- `model`: architecture of the Q function. Depth of conv layers, filter widths, etc.
- `algo`: hyperparameters in Q learning. Learning rate, discounting factor, Q-target update frequency, etc.
- `replay`: replay memory settings. Memory size, priority schedule, eviction policy, etc. 

### 2. env_config

- `action_spec`: provides information to the model builder. For Gym envs, `action_spec` is redundant, but must be consistent with their `action_space`. 
    - `type`: `continuous` or `discrete`. 
    - `dim`: a list that specifies the action shape in continuous control, or a singleton list of the action dimension in discrete tasks. 
   
- `obs_spec`: provides information to the model builder. For Gym envs, `obs_spec` is redundant, but must be consistent with their `obs_space`. 
    - `type`: TODO
    
    
### 3. session_config

Configures the cluster, monitor, evaluator, Tensorplex, and logger. Takes care of all book-keeping. It will have at least the following keys/sections:

- `folder`: root folder that stores everything about the experiment run, including Tensorboard files, logs, and checkpoints.
- `replay`: replay server specs. In v0.2, replay server is a _standalone process_ independent from learner.
- `ps`: parameter server specs.
- `tensorplex`: distributed Tensorboard, see [Tensorplex](https://github.com/stanfordvl/tensorplex).
	- `update_schedule`: Tensorplex update periods in various components.
- `loggerplex`: distributed logging, see [Tensorplex](https://github.com/stanfordvl/tensorplex).
- `sender`: agent side calls `env.ExpSenderWrapper` to send experience tuples to the replay server. Sender config includes `flush_iteration`, `flush_time`, etc. 

Please refer to `session.default_configs.BASE_SESSION_CONFIG`.


# Installation

## Main surreal library

```
git clone https://github.com/StanfordVL/Surreal.git
pip install -e Surreal/
```

## Tensorplex

Read about [Tensorplex API on Github](https://github.com/StanfordVL/Tensorplex).

```
git clone https://github.com/StanfordVL/Tensorplex.git
pip install -e Tensorplex/
```

## MujocoManipulation
Follow the instructions in [MujocoManipulation](https://github.com/StanfordVL/MujocoManipulation)

## Docker and Kubernetes

TODO
