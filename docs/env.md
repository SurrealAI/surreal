# Environments

Environments are created and run on agent machines; the API methods they expose
are detailed below.

# Env initialization
You can find examples of environment initialization in surreal/env/make_env.py.
The make_env function is used in surreal/agent/base.py. In addition, note the lines

```python
env_config.action_spec = env.action_spec()
env_config.obs_spec = env.observation_spec()
```
    
in surreal/env/make_env.py.  These lines populate the env_config with the action
and observation specs, which specify the dimensions of the action and observation
space.  This means that the action and observation dimensions are generated at
runtime rather than specified manually in the env_config.

# Available envs

The surreal repository contains several environments available for use.  If you want
to create your own environment class, see the section
[Building your own custom environments](#building-your-own-custom-environments).
Note that some of the environments listed below may require a mujoco license.

- OpenAI gym environments, e.g. `make_env(gym:HalfCheetah-v2)`
- Deepmind control suite environments, e.g. `make_env('dm_control:humanoid-walk')`
- Surreal Robotics Suite environments, e.g. `make_env(robosuite:SawyerLift)`

# Environment wrappers
Environment wrappers are environments which wrap around an environment and
applies some modification. For example, we have environments which will perform
frame stacking or grayscaling.

In wrapper.py, you will find wrappers that take environments and reformat them into the format that the models expect.

For example, the command below will create a dm_control environment.

`suite.load(domain_name=domain_name, task_name=task_name, visualize_reward=record_video)`

This environment has an observation spec that looks like

`collections.OrderedDict([('pixels', dm_control.rl.specs.ArraySpec(shape=(84, 84, 3), dtype=np.dtype('float32')))])`

We reformat the above observation spec so that it satisfies the Surreal format. The observation format we use is as follows:

## observation (OrderedDict):
observation dictionaries are returned in both the `env.step()` and `env.reset()` functions

`type(obs[modality][name]) == nparray`, e.g.

`obs[modality][name].shape == (3, 84, 84)` for an 84 x 84 rgb image

Here, `modality` refers to the high-level type of the observation.  The modalities we use are 'pixel' for pixel observations
and 'low_dim' for low_dimensional observations.  We may have a future modality type for force readings.

The reason we index by modality first is so that the model knows what kind of network architecture to apply for each
observation. We apply convolution networks to `pixel` modality and mlp networks to `low_dim` observations.

The 'name' key simply refers to the name of the observation, such as 'camera0', 'camera1', 'position', or 'velocity'.
Many environments use the `ObservationConcatenationWrapper`, which takes all observations under the `low_dim` modality and
concatenates them into the new array at `obs['low_dim']['flat_inputs']`.

Example observation:

```
{
    'pixel': {
        'camera0': (numpy array),
    },
    'low_dim': {
        'position': (numpy array),
        'velocity': (numpy array),
    }
}
```

## observation_spec (OrderedDict):
observation_spec dictionaries are returned by the `env.observation_spec()` function

`type(obs_spec[modality][name]) == tuple`, e.g.

`obs_spec[modality][name] == (3, 84, 84)` for an 84 x 84 rgb image

Note that the keys for observation and observation_spec are the same,
so `obs_spec[modality][name] == obs[modality][name].shape` should always be true

## action_spec (Dict)
action_spec dictionaries are returned by the `env.action_spec()` function

action_spec has the format of type

```
{
    'type': (ActionType)
    'dim': (tuple)
}
```

e.g.

```
{
    'type': ActionType.continuous,
    'dim': (7,)
}
```

# Building Your Own Custom Environments
To create a new environment, you should subclass the class Env in surreal/env/base.py.
Your env must override the necessary methods listed in that class in order for your env
to work.  Finally, you should add a corresponding entry into surreal/env/make_env.py
so that make_env('your_env_name') will work.