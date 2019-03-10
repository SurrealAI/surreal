import torch
import torch.nn as nn
import torch.nn.functional as F
import torchx as tx
from torchx.nn.hyper_scheduler import *
import numpy as np
from .ppo import PPOLearner
from surreal.model.gail_net import GAILModel
from surreal.env import make_env

class GAILLearner(PPOLearner):
    '''
    GAILLearner: subclass of PPOLearner that contains GAIL algorithm logic
    Attributes:
        gpu_option: 'cpu' if not using GPU, 'cuda:all' otherwise
        model: instance of PPOModel from surreal.model.ppo_net
        ref_target_model: instance of PPOModel, kept to used as
            reference policy
        ppo_mode: string of either 'adapt' or 'clip' to determine
            which variant of PPO is used. For details of variants
            see https://arxiv.org/pdf/1707.06347.pdf
        norm_adv: boolean flag -- whether to use batch advantage
            normalization
        use_z_filter: boolean flag -- whether to use obs Z-Filtering
        actor/critic_optim: Adam Optimizer for policy and baseline network
        actor/critic_lr_scheduler: Learning rate scheduler. details see
            surreal.utils.pytorch.scheduler
        aggregator: experience aggregator used to batch experiences.
            for available aggregators, see surreal.learner.aggregator
        pd: probability distribution class (Assumed as Diagonal Gaussian)
            see surreal.model.ppo_net for details

    important member functions:
        private methods:
        _clip_loss: computes the loss and various statistics
            for 'clip' variant PPO
        _clip_update: uses loss information to make policy update
        _adapt_loss: computes loss and various statistics for
            'adapt' variant of PPO
        _adapt_update: uses loss info to make policy update
        _value_loss: computes loss and various statistics for value function
        _value_update: uses loss info to update value function
        _gae_and_return: computes generalized advantage estimate and
            corresponding N-step return. Details of algorithm can be found
            here: https://arxiv.org/pdf/1506.02438.pdf
        _advantage_and_return: basic advantage and N-step return estimate
        _optimize: fucntion that makes policy and value function update
        _post_publish: function that manages metrics and behavior after
            parameter release

        public methods:
        learn: method to perform optimization and send to tensorplex for log
        module_dict: returns the corresponding parameters
        publish_parameter: publishes parameters in self.model to parameter server
    '''
    def __init__(self, learner_config, env_config, session_config):
        # PPO setup
        super().__init__(learner_config, env_config, session_config)

        # GAIL-specific setup
        self.reward_lambda = self.learner_config.algo.reward_lambda # reward mixing
        self.lr_discriminator = self.learner_config.algo.network.lr_discriminator
        self.epoch_discriminator = self.learner_config.algo.consts.epoch_discriminator
        self.stride = self.learner_config.algo.stride

        # learning rate setting:
        num_updates = int(self.frames_to_anneal / self.learner_config.parameter_publish.exp_interval)
        lr_scheduler = eval(self.learner_config.algo.network.anneal.lr_scheduler)

        with tx.device_scope(self.gpu_option):
            # TODO: what hypers does GAIL need? put them here ###
            # add a discriminator
            self.discriminator_model = GAILModel(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                model_config=self.learner_config.model,
                use_cuda=self.use_cuda,
                use_z_filter=self.use_z_filter
            )

            # Learning parameters and optimizer
            self.clip_discriminator_gradient = self.learner_config.algo.network.clip_discriminator_gradient
            self.discriminator_gradient_clip_value = self.learner_config.algo.network.discriminator_gradient_norm_clip

            self.discriminator_optim = torch.optim.Adam(
                self.discriminator_model.get_discriminator_params(),
                lr=self.lr_discriminator,
                weight_decay=self.learner_config.algo.network.discriminator_regularization
            )

            # learning rate scheduler
            self.discriminator_lr_scheduler  = lr_scheduler(self.discriminator_optim, 
                                                            num_updates,
                                                            update_freq=self.lr_update_frequency,
                                                            min_lr = self.min_lr)

            # discriminator loss
            self.discriminator_loss = nn.BCEWithLogitsLoss()

        self._load_demo_sampler(env_config)
        self.iteration = 0
        self.freeze = 50

    def _load_demo_sampler(self, env_config):
        assert env_config.demonstration is not None, "need demo_config set in env_config for discriminator"
        self.demo_env, config = make_env(env_config)
        self.demo_env.use_camera_obs = False
        assert self.demo_env.demo_sampler is not None
        self.demo_env.demo_sampler.need_xml = False

    # def module_dict(self):
    #     '''
    #     returns the corresponding parameters (overrides PPO's module_dict)
    #     '''
    #     return {
    #         'gail': self.model,
    #     }

    def _preprocess_batch_ppo(self, batch):
        '''
            overrides PPO implementation and extracts GAIL rewards from persistent_infos

            Loading experiences from numpy to torch.FloatTensor type
            Args: 
                batch: BeneDict of experiences containing following attributes
                        'obs' - observation
                        'actions' - actions
                        'rewards' - rewards
                        'obs_next' - next observation
                        'persistent_infos' - action policy
                        'onetime_infos' - RNN hidden cells or None
            Return:
                Benedict of torch.FloatTensors
        '''

        ### TODO: log correct stats... ###
        ### TODO: extract object features before making a tensor here... ###

        # We compute gail rewards per subtrajectory via reshaping
        import time
        t = time.time()
        # get GAIL rewards for this batch
        with tx.device_scope(self.gpu_option):
            with torch.no_grad():
                obs = batch['obs'] # each observation has value of shape (batch_size, horizon, obs_dim)
                obs = obs["low_dim"]["flat_inputs"]
                action = batch["actions"]

                _, horizon, obs_dim = obs.shape
                _, _, action_dim = action.shape

                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_tensor = torch.tensor(action, dtype=torch.float32)

                combined = torch.cat((obs_tensor, action_tensor), dim=2)
                obs_action_tensor = combined.view(-1, obs_dim + action_dim)

                # compute rewards over entire subtrajectory
                # in one forward pass, then reshape
                gail_rewards = (
                   self.discriminator_model.get_discriminator_reward(
                        obs_action_tensor)
                )
                gail_rewards = torch.clamp(gail_rewards, max=10.).cpu().numpy()
                gail_rewards = gail_rewards.reshape(-1, horizon)

        print("time taken to compute gail_rewards: {}".format(time.time() - t))

        # compute mixed rewards and feed it to PPO
        env_rewards = np.array(batch['rewards'])
        batch['rewards'] = self.reward_lambda * gail_rewards + (1. - self.reward_lambda) * env_rewards
        data_file = open("/Users/peter/disc_update_file.txt", "a")
        data_file.write(str(np.mean(gail_rewards)) + "\n")
        preproc_batch = super()._preprocess_batch_ppo(batch)
        # keep track of pure env and pure gail rewards as well
        with tx.device_scope(self.gpu_option):
            preproc_batch['env_rewards'] = torch.tensor(env_rewards, dtype=torch.float32)
            preproc_batch['gail_rewards'] = torch.tensor(gail_rewards, dtype=torch.float32)
            return preproc_batch

    def _get_demo_samples(self, N):
        out = []
        for i in range(N):
            obs, action = self.demo_env.demo_sampler._uniform_sample()
            # TODO refactor the matryoshka env/wrappers
            #obs = self.demo_env._flatten_obs(self.demo_env._filtered_obs(self.demo_env._add_modality(obs)))
            #for mod in obs.keys():
            #    if not out.get(mod):
            #        out[mod] = {}
            #    for k in obs[mod].keys():
            #        if not out[mod].get(k):
            #            out[mod][k] = []
            #        out[mod][k].append(obs[mod][k])
            out.append(np.concatenate((obs, action), axis=1))

        #obs_tensor = {}
        #for mod in obs.keys():
        #    obs_tensor[mod] = {}
        #    for k in obs[mod].keys():
        #        obs_tensor[mod][k] = torch.tensor(out[mod][k], dtype=torch.float32)

        obs_tensor = torch.tensor(out, dtype=torch.float32)
        return obs_tensor

    def _discriminator_loss(self, actor_obs, expert_obs):
        # forward passes for logits
        actor_logits = self.discriminator_model.forward_discriminator(actor_obs)
        expert_logits = self.discriminator_model.forward_discriminator(expert_obs)

        # accuracies
        actor_acc = (actor_logits < 0).float().mean()
        expert_acc = (expert_logits > 0).float().mean()

        # losses
        actor_loss = self.discriminator_loss(actor_logits, torch.zeros_like(actor_logits))
        expert_loss = self.discriminator_loss(expert_logits, torch.ones_like(expert_logits))
        loss = actor_loss + expert_loss

        stats = {
            'discriminator_loss': loss.item(),
            'discriminator_actor_loss': actor_loss.item(),
            'discriminator_expert_loss': expert_loss.item(),
            'discriminator_actor_accuracy': actor_acc.item(),
            'discriminator_expert_accuracy': expert_acc.item()
        }

        return loss, stats

    def _discriminator_update(self, actor_obs, expert_obs):
        loss, stats = self._discriminator_loss(actor_obs, expert_obs)
        self.discriminator_model.clear_discriminator_grad()
        loss.backward()
        if self.clip_discriminator_gradient:
            stats['grad_norm_discriminator'] = nn.utils.clip_grad_norm_(
                                                self.discriminator_model.get_discriminator_params(), 
                                                self.discriminator_gradient_clip_value)
        self.discriminator_optim.step()
        return stats

    def _optimize(self, batch):
        '''
            main method for optimization that calls _adapt/clip_update and 
            _value_update epoch_policy and epoch_baseline times respectively
            return: dictionary of tracted statistics
            Args:
                batch: Benedict of torch.FloatTensors with the following keys:
                    obs: batch of observations (batch_size, N-step , obs_dim)
                    obs_next: batch of next observations (batch_size, 1 , obs_dim)
                    actions: batch of actions (batch_size, N-step , act_dim)
                    rewards: batch of rewards (batch_size, N-step)
                    dones: batch of termination flags (batch_size, N-step)
                    action_infos: list of batched other attributes tracted, such as
                        behavior policy, RNN hidden states and etc.
            Returns:
                dictionary of recorded statistics

        '''

        # update policy with PPO
        if self.iteration > self.freeze:
            stats = super()._optimize(batch)
        else:
            stats = {}

        # update GAIL discriminator
        with tx.device_scope(self.gpu_option):

            ### TODO: should we split this into smaller batches? ###
            # split observations over multiple timesteps into independent observations
            # according to the stride, to make sure they are not duplicated.
            # this effectively increases the batch size for the discriminator
            # TODO(peter): double check stride logic
            obs = batch.obs["low_dim"]["flat_inputs"]
            actions = batch["actions"]
            bsize = obs.size()[0]
            if self.if_rnn_policy:
                obs = obs[:, :self.stride, :].contiguous().view(bsize * self.stride, -1).detach()
                actions = actions[:, :self.stride, :].contiguous().view(bsize * self.stride, -1).detach()
            else:
                obs = obs[:, 0, :].contiguous().detach()
                actions = actions[:, 0, :].contiguous().detach()

            obs = torch.cat((obs, actions), dim=1)

            #obs_iter = {}
            #for mod in obs.keys():
            #    obs_iter[mod] = {}
            #    for k in obs[mod].keys():
            #        if self.if_rnn_policy:
            #            bsize = obs[mod][k].size()[0]
            #            obs_iter[mod][k] = obs[mod][k][:, :self.stride, :].contiguous().view(bsize * self.stride, -1).detach()
            #        else:
            #            obs_iter[mod][k] = obs[mod][k][:, 0, :].contiguous().detach()
            expert_obs = torch.squeeze(self._get_demo_samples(bsize * self.stride))
            #for _ in range(self.epoch_discriminator):
            self.iteration += 1
            print(self.iteration)
            if self.iteration % self.epoch_discriminator == 0:
                discriminator_stats = self._discriminator_update(actor_obs=obs, expert_obs=expert_obs)

                # log stats
                for k in discriminator_stats:
                    stats[k] = discriminator_stats[k]
            stats["env_rewards"] = batch.env_rewards.mean().item()
            stats["gail_rewards"] = batch.gail_rewards.mean().item()
            print("env_rewards: {}".format(stats["env_rewards"]))
            print("gail_rewards: {}".format(stats["gail_rewards"]))

            return stats

    def checkpoint_attributes(self):
        '''
            outlines attributes to be checkpointed
        '''

        # add discriminator model to list of checkpointed attributes
        attributes = super().checkpoint_attributes()
        attributes.append('discriminator_model')
        return attributes

        # return [
        #     'model',
        #     'ref_target_model',
        #     'actor_lr_scheduler',
        #     'critic_lr_scheduler',
        #     'current_iteration',
        # ]


