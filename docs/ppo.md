# Tutorial for Surreal PPO

In this tutorial, we will go through the Proximal Policy Gradient algorithm in this specific implementation. In the following sections, we will introduce the algorithm, discuss some of its key features, outline the expected behaviors in terms of some metrics, and explain the configuration file in details.

## PPO

PPO in its core is a soft trust region policy optimization approach. The classic TRPO approach by John Schulman formulates the problem by maximizing likelihood of actions subjected to the change in policy (KL divergence) must of lower than a certain threshold value.

PPO, on the other hand, takes a soft approach: namely the algorithm is encouraged to make update that lies within the “trust region” that is at most delta away from the original policy in terms of KL-divergence. Although KL divergence is not a strict distance metric theoretically. In this document, it will generally be indicative of how far policies are away from each other.

There are two major variants of PPO at the moment, one by OpenAI and the other by DeepMind. The [first one](https://arxiv.org/pdf/1707.06347.pdf), commonly refers to `clip` mode, pessimistically clips the probability ratio (can be seen as important sampling weights). The [other one](https://arxiv.org/pdf/1707.02286.pdf), commonly referred to as `adapt`, adaptively penalizes the KL divergence between the old and new policy (imagine the KL-penalty as a regularization term).

## Important Features of Surreal PPO:
PPO in its vanilla form is an on-policy algorithm, which requires the learner's learned distribution to be the same as the one agent used to generate the data. However, in distributed setting such as surreal, we cannot guarantee such synchronization. Thus, we applied a few modification to the algorithm to stabilize and improve the performance.

* **Target Network**: Due to our design's asynchronous nature and inevitable network latency, the actors' behavior policy that generates the experience trajectories can lag behind the learner's policy by several updates at the time of gradient computation. To address this issue, PPO learner keeps a target network that is broadcast to all actors at a lower frequency. This ensures that a much larger portion of the experience trajectories are on-policy, except for those generated within the policy lag. Empirically, we find that the target network mechanism reduces off-policyness and improves PPO's performance significantly.
* **Observation Z-filtering**: we keep a running mean and variance of the observations. At each iteration, we pass the raw observation into this z-filter first to get a zero mean unit variance whitened observation, which is then inputted to the actor/critic network. The running mean and variance is released at the same time with the actor/critic network parameters.
Despite the effectiveness of z-filtering on low dimensional input, z-filtering does not work with pixel based inputs (at least on HalfCheetah). This is because the camera is centered on the agent itself. After z-filtering, the input image will be black.On the note of normalization, Layernorm and Batchnorm are briefly considered to replace z-filter. However, these are actually both not ideal: 

* **LSTM**: instead of taking input only the observation at a timestep, our implementation also take input a tuple of hidden states from previous timesteps using a LSTM. Noticeably, the idea of horizon is introduced. Horizon is used as the depth of generalized advantage estimate for advantage calculation for every data point in a truncated rollout. We found that LSTM when used with action repeat is highly benefitial for longer horizon tasks such as our robotic benchmark.

* Other modifcation includes using generalized advantage actor critic (details [here](https://arxiv.org/pdf/1506.02438.pdf) and batch advantage normalization. For full implementation detail and pseudocode, please see manuscript.

## Key Metrics and their Expected Behaviors 

Key Metrics:
*  `_pol_kl`: KL divergence between target and new policy. This should fluctuate and should not exceed `2 * kl_target`. If it does, a large penalty is incurred in `adapt` method. You should expect a dip in performance
    
*  `_avg_return_target`: highly correlated with the performance of the agents. This metric denotes the average N-step return for each update. You can see when the agents stop learning by looking at this metric
    
*  `_ref_behave_diff`: tracks how different is the target network and the policy used to generate the data. With target network, this should be in range of `1e-4` or lower.
    
*  `_avg_log_sig`, `entropy`: these two will tell you the state of your algorithm. If `_avg_log_sig` is increasing, that means the algorithm is zoning in on a specific policy. They should generally decrease as training progresses if the agent is learning useful information. You may see in the first iterations it may increase. But in long term they should decrease.

## Implementation detail

The code for Surreal PPO are contained in the following files
* [Custom PPO Learner class](../surreal/learner/ppo.py)
* [Custom PPO Agent class](../surreal/agent/ppo_agent.py)
* [Custom PPO Model class](../surreal/model/ppo_net.py)
* [PPO network architectures](../surreal/model/model_builders/builders.py) (Note: we use TorchX for our networks)
* [PPO configurations](../surreal/main/ppo_configs.py)

In the configuration file, there are many many hyperparameters. In the following section, we will go through each section in detail. For each algorithm, we have 3 nested dictionary configs that separates [learner](../surreal/main/ppo_configs.py#L20), [environment](../surreal/main/ppo_configs.py#L103), and [session](../surreal/main/ppo_configs.py#L130) configurations. Particularly for PPO, there are many relevant parameters. In this section we will explain in details the most important ones.

Learner Config:
* `learner_config.model`: contains model architecture design such as `actor_fc_hidden_sizes` and `cnn_feature_dim`
* `learner_config.algo.network`: contains optimizer, learning rate, and annealing setup. Note that `network.lr_scheduler` should specify a class in either `torch.optim.lr_scheduler` or `torchx.nn.hyper_scheduler`
* `learner_config.algo.rnn`: specifies LSTM layers and hidden units
* `learner_config.algo.consts`: specifies some training constants, such as initial log standard deviation `consts.init_log_sig`and target KL divergence for each parameter release `consts.kl_target` 
* `learner_config.algo.adapt_consts`: specifies hyperparameters specifically for `adapt` PPO. Important hyperparameters include `adapt_consts.kl_cutoff_coeff` which is the coefficient for KL penalty when the KL divergence of update exceeds twice the target KL divergence
* `learner_config.replay`: specifies the replay buffer used. For PPO we use `FIFOQueue` for `replay.replay_class` and small queue length `replay.memory_size`. In this case we use 96.
* `learner_config.parameter_publish.exp_interval`: specifies how often learner pushes parameter to Parameter Server. 4096 denotes the number of sub-trajectory processed until parameter is published. With batch size of 64, we publish parameters every `4096/64=64` mini-batches.

Environment Config:
* `env_config.action_repeat`: specifies how many times the input action is repeated before allowing next action input from actor. We find this highly impactful for Robotic Manipulation benchmark tasks
* `env_config.stochastic_eval`: specifies whether agent uses deterministic (mean) or stochastic (mean sampled with standard deviation) policy for evaluation.
* `env_config.demonstration`: specifies setup for sampling states from collected expert demonstration
	* `demonstration.adaptive`: whether we use adaptive or open-loop curriculum. In adaptive setup, the curriculum updates only if certain improvement criteria is met, whereas in open-loop curriculum setup, curriculum is updated based on fixed episode interval. Please read our manuscript and appendix for full details
	* `demonstration.mixing` and `demonstration.mixing_ratio`: `demonstration.mixing` specifies which curricula of `random`, `uniform`, `forward`, `reverse` we want to use and `demonstration.mixing_ratio` specifies how often each should be used. Note that the values in `demonstration.mixing_ratio` must sum to `1.0`

Session Config:
* `session_config.agent.fetch_parameter_mode`: specifies how should the agent poll parameters from Parameter Server. The choice is either `step` (every certain steps) or `episode` (every certain episodes)
* `session_config.agent.fetch_parameter_interval`: specifies how often agents would poll parameter server for new parameters. If no new parameters are available, this will be no_op.
* `session_config.checkpoint`: specifies the interval for checkpointing models. 