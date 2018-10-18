# Tutorial for Surreal DDPG

In this tutorial, we will go through the Surreal implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm. In the following sections, we will introduce the algorithm, discuss some of its key features, outline the expected behaviors in terms of some metrics, and explain the configuration file in detail.

## DDPG

DDPG is a reinforcement learning algorithm adapted to be successful on continuous control environments.  It adopts inspiration from DQN, specifically the target network and replay buffer.

The target network is a representation of a previous iteration of the model, and is used to do the value estimation of the next state in the Bellman equation. We update the target network using either soft or hard updates. In soft update mode, after each gradient update on the  main network, we step the parameters of the target network in the direction of the main network by a small fraction τ. In hard update mode, after a given number of iterations, we set the target network parameters to be equal to the current network parameters.

The replay buffer performs a similar function of mitigating the moving target problem.  Recent experiences are stored in a buffer, and batches are sampled uniformly at random from the buffer to perform gradient descent on instead of immediately performing gradient descent on experiences after they are generated.  This slows down the rate at which the network has to model the value of a shifting policy.

## Implementation details

The code for Surreal DDPG is contained in the following files
* [DDPG Learner class](../surreal/learner/ddpg.py)
* [DDPG Agent class](../surreal/agent/ddpg_agent.py)
* [DDPG Model class](../surreal/model/ddpg_net.py)
* [DDPG network architectures](../surreal/model/model_builders/builders.py) (Note: we use TorchX for our networks)
* [DDPG parameters](../surreal/main/ddpg_configs.py)

For each algorithm, we have 3 nested dictionary configs that separates [learner](../surreal/main/ddpg_configs.py#L20), [environment](../surreal/main/ddpg_configs.py), and [session](../surreal/main/ddpg_configs.py) configurations.

Learner Config:
* `learner_config.model`: contains model architecture design such as `actor_fc_hidden_sizes` and `cnn_feature_dim`
* `learner_config.algo.network`: contains parameter update parameters, including learning rate, target_network_update, and weight regularization.
* `learner_config.replay`: specifies the replay buffer used. For DDPG this should be `UniformReplay` for `replay.replay_class`. This replay is sharded, and by default the sum of the sharded memories is 1,000,000 experiences.
* `learner_config.parameter_publish.exp_interval`: specifies how often learner pushes parameter to Parameter Server. For ddpg, parameter publish is time-based, and occurs at set time intervals.

Environment Config:
* See the [Environment documentations](env.md) for details on observation and action formats.
* `env_config.action_repeat`: specifies how many times the input action is repeated before allowing next action input from actor. We find this highly impactful for Robotic Manipulation benchmark tasks
* `env_config.pixel_input`: set to True if the environment returns image output.
* `env_config.limit_episode length`: specifies the maximum number of steps an environment can perform before termination.

Session Config:
* `session_config.tensorplex`: specifies how often tensorplex will record statistics about learning progression.
* `session_config.agent.fetch_parameter_mode`: specifies how should the agent poll parameters from Parameter Server. The choice is either `step` (every certain steps) or `episode` (every certain episodes)
* `session_config.agent.fetch_parameter_interval`: specifies how often agents would poll parameter server for new parameters. If no new parameters are available, this will be no_op.
