# Environment wrappers
In wrapper.py, you will find wrappers that take environments and reformat them into the format that the models expect.

For example, 

`suite.load(domain_name=domain_name, task_name=task_name, visualize_reward=record_video)`

will return an environment whose observation spec looks like

`collections.OrderedDict([('pixels', dm_control.rl.specs.ArraySpec(shape=(84, 84, 3), dtype=np.dtype('float32')))])`

The observation format we use is as follows:

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