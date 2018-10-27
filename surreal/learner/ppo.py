import torch
import torch.nn as nn
import torchx as tx
from torchx.nn.hyper_scheduler import *
import numpy as np
from .base import Learner
from .aggregator import MultistepAggregatorWithInfo 
from surreal.model.ppo_net import PPOModel, DiagGauss
from surreal.model.reward_filter import RewardFilter
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG, ConfigError

class PPOLearner(Learner):
    '''
    PPOLearner: subclass of Learner that contains PPO algorithm logic
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
        super().__init__(learner_config, env_config, session_config)

        # GPU setting
        self.current_iteration = 0
        self.global_step = 0
        if not torch.cuda.is_available():
            self.gpu_option = 'cpu'
        else:
            self.gpu_option = 'cuda:all'
        self.use_cuda = torch.cuda.is_available()

        if not self.use_cuda:
            self.log.info('Using CPU')
        else:
            self.log.info('Using GPU: {}'.format(self.gpu_option)) 

        # RL general parameters
        self.gamma = self.learner_config.algo.gamma
        self.lam   = self.learner_config.algo.advantage.lam
        self.n_step = self.learner_config.algo.n_step
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.use_r_filter = self.learner_config.algo.use_r_filter
        self.norm_adv = self.learner_config.algo.advantage.norm_adv
        self.batch_size = self.learner_config.replay.batch_size

        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_spec = self.env_config.obs_spec                                           
        self.init_log_sig = self.learner_config.algo.consts.init_log_sig

        # PPO parameters
        self.ppo_mode = self.learner_config.algo.ppo_mode
        self.if_rnn_policy = self.learner_config.algo.rnn.if_rnn_policy
        self.horizon = self.learner_config.algo.rnn.horizon
        self.lr_actor = self.learner_config.algo.network.lr_actor
        self.lr_critic = self.learner_config.algo.network.lr_critic
        self.epoch_policy = self.learner_config.algo.consts.epoch_policy
        self.epoch_baseline = self.learner_config.algo.consts.epoch_baseline
        self.kl_target = self.learner_config.algo.consts.kl_target
        self.adjust_threshold = self.learner_config.algo.consts.adjust_threshold
        self.reward_scale = self.learner_config.algo.advantage.reward_scale

        # PPO mode 'adjust'
        self.kl_cutoff_coeff = self.learner_config.algo.adapt_consts.kl_cutoff_coeff
        self.beta_init = self.learner_config.algo.adapt_consts.beta_init
        self.beta_range = self.learner_config.algo.adapt_consts.beta_range

        # PPO mode 'clip'
        self.clip_range = self.learner_config.algo.clip_consts.clip_range
        self.clip_epsilon_init = self.learner_config.algo.clip_consts.clip_epsilon_init

        if self.ppo_mode == 'adapt':
            self.beta = self.beta_init
            self.eta = self.kl_cutoff_coeff
            self.beta_upper = self.beta_range[1]
            self.beta_lower = self.beta_range[0]
            self.beta_adjust_threshold = self.adjust_threshold
        else: # method == 'clip'
            self.clip_epsilon = self.clip_epsilon_init
            self.clip_adjust_threshold = self.adjust_threshold
            self.clip_upper = self.clip_range[1]
            self.clip_lower = self.clip_range[0]

        # learning rate setting:
        self.min_lr = self.learner_config.algo.network.anneal.min_lr
        self.lr_update_frequency = self.learner_config.algo.network.anneal.lr_update_frequency
        self.frames_to_anneal = self.learner_config.algo.network.anneal.frames_to_anneal
        num_updates = int(self.frames_to_anneal / self.learner_config.parameter_publish.exp_interval)
        lr_scheduler = eval(self.learner_config.algo.network.anneal.lr_scheduler) 

        self.exp_counter = 0
        self.kl_record = []

        with tx.device_scope(self.gpu_option):
            self.model = PPOModel(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                model_config=self.learner_config.model,
                use_cuda=self.use_cuda,
                init_log_sig=self.init_log_sig,
                use_z_filter=self.use_z_filter,
                if_pixel_input=self.env_config.pixel_input,
                rnn_config=self.learner_config.algo.rnn,
            )
            self.ref_target_model = PPOModel(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                model_config=self.learner_config.model,
                use_cuda=self.use_cuda,
                init_log_sig=self.init_log_sig,
                use_z_filter=self.use_z_filter,
                if_pixel_input=self.env_config.pixel_input,
                rnn_config=self.learner_config.algo.rnn,
            )
            self.ref_target_model.update_target_params(self.model)

            # Learning parameters and optimizer
            self.clip_actor_gradient = self.learner_config.algo.network.clip_actor_gradient
            self.actor_gradient_clip_value = self.learner_config.algo.network.actor_gradient_norm_clip
            self.clip_critic_gradient = self.learner_config.algo.network.clip_critic_gradient
            self.critic_gradient_clip_value = self.learner_config.algo.network.critic_gradient_norm_clip

            self.critic_optim = torch.optim.Adam(
                self.model.get_critic_params(),
                lr=self.lr_critic,
                weight_decay=self.learner_config.algo.network.critic_regularization
            )
            self.actor_optim = torch.optim.Adam(
                self.model.get_actor_params(),
                lr=self.lr_actor,
                weight_decay=self.learner_config.algo.network.actor_regularization
            )

            # learning rate scheduler
            self.actor_lr_scheduler  = lr_scheduler(self.actor_optim, 
                                                    num_updates,
                                                    update_freq=self.lr_update_frequency,
                                                    min_lr = self.min_lr)
            self.critic_lr_scheduler = lr_scheduler(self.critic_optim, 
                                                    num_updates,
                                                    update_freq=self.lr_update_frequency,
                                                    min_lr = self.min_lr)

            # Experience Aggregator
            self.aggregator = MultistepAggregatorWithInfo(self.env_config.obs_spec, 
                                                          self.env_config.action_spec)
        
            # probability distribution. Gaussian only for now
            self.pd = DiagGauss(self.action_dim)

            # placeholder for RNN hidden cells
            self.cells = None

            # Reward White-filtering
            if self.use_r_filter: 
                self.reward_filter= RewardFilter()

    def _clip_loss(self, obs, actions, advantages, behave_pol): 
        """
        Computes the loss with current data. also returns a dictionary of statistics
        which includes surrogate loss, clipped surrogate los, policy entropy, clip
        constant
        return: surreal.utils.pytorch.GPUVariable, dict
        Args:
            obs: batch of observations in form of (batch_size, obs_dim)
            actions: batch of actions in form of (batch_size, act_dim)
            advantages: batch of normalized advantage, (batch_size, 1)
            behave_pol: batch of behavior policy (batch_size, 2 * act_dim)
        Returns:
            clip_loss: Variable for loss
            stats: dictionary of recorded statistics
        """
        learn_pol = self.model.forward_actor(obs, self.cells)
        learn_prob = self.pd.likelihood(actions, learn_pol)
        behave_prob = self.pd.likelihood(actions, behave_pol)
        prob_ratio = learn_prob / behave_prob
        cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 
                                               1 + self.clip_epsilon)
        surr = -prob_ratio * advantages.view(-1, 1)
        cliped_surr = -cliped_ratio * advantages.view(-1, 1)
        clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean() 

        stats = {
            "_surr_loss": surr.mean().item(), 
            "_clip_surr_loss": clip_loss.item(), 
            "_entropy": self.pd.entropy(learn_pol).mean().item(),
            '_clip_epsilon': self.clip_epsilon
        }
        return clip_loss, stats

    def _clip_update(self, obs, actions, advantages, behave_pol):
        """
        Method that makes policy updates. calls _clip_loss method
        Note:  self.clip_actor_gradient determines whether gradient is clipped
        return: dictionary of statistics to be sent to tensorplex server
        Args:
            obs: batch of observations in form of (batch_size, obs_dim)
            actions: batch of actions in form of (batch_size, act_dim)
            advantages: batch of normalized advantage, (batch_size, 1)
            behave_pol: batch of behavior policy (batch_size, 2 * act_dim)
        Returns:
            stats: dictionary of recorded statistics
        """
        loss, stats = self._clip_loss(obs, actions, advantages, behave_pol)
        self.model.clear_actor_grad()
        loss.backward()
        if self.clip_actor_gradient:
            stats['grad_norm_actor'] = nn.utils.clip_grad_norm_(
                                            self.model.get_actor_params(), 
                                            self.actor_gradient_clip_value).item()
        self.actor_optim.step()
        return stats

    def _adapt_loss(self, obs, actions, advantages, behave_pol, ref_pol):
        """
        Computes the loss with current data. also returns a dictionary of statistics
        which includes surrogate loss, clipped surrogate los, policy entropy, adaptive
        KL penalty constant, policy KL divergence
        return: surreal.utils.pytorch.GPUVariable, dict
        Args:
            obs: batch of observations in form of (batch_size, obs_dim)
            actions: batch of actions in form of (batch_size, act_dim)
            advantages: batch of normalized advantage, (batch_size, 1)
            behave_pol: batch of behavior policy (batch_size, 2 * act_dim)
            ref_pol: batch of reference policy (batch_size, 2 * act_dim)
        Returns:
            loss: Variable for loss
            stats: dictionary of recorded statistics
        """
        learn_pol = self.model.forward_actor(obs, self.cells)
        prob_behave = self.pd.likelihood(actions, behave_pol)
        prob_learn  = self.pd.likelihood(actions, learn_pol)
        
        kl = self.pd.kl(ref_pol, learn_pol).mean()
        surr = -(advantages.view(-1, 1) * (prob_learn/ torch.clamp(prob_behave, min=1e-2))).mean()
        loss = surr + self.beta * kl
        entropy = self.pd.entropy(learn_pol).mean()

        if kl.item() - 2.0 * self.kl_target > 0:
            loss += self.eta * (kl - 2.0 * self.kl_target).pow(2)

        stats = {
            '_kl_loss_adapt': loss.item(), 
            '_surr_loss': surr.item(), 
            '_pol_kl': kl.item(), 
            '_entropy': entropy.item(),
            '_beta': self.beta
        }
        return loss, stats

    def _adapt_update(self, obs, actions, advantages, behave_pol, ref_pol):
        """
        Method that makes policy updates. calls _adapt_loss method
        Note:  self.clip_actor_gradient determines whether gradient is clipped
        return: dictionary of statistics to be sent to tensorplex server
        Args:
            obs: batch of observations in form of (batch_size, obs_dim)
            actions: batch of actions in form of (batch_size, act_dim)
            advantages: batch of normalized advantage, (batch_size, 1)
            behave_pol: batch of behavior policy (batch_size, 2 * act_dim)
            ref_pol: batch of reference policy (batch_size, 2 * act_dim)
        Returns:
            stats: dictionary of recorded statistics
        """
        loss, stats = self._adapt_loss(obs, actions, advantages, behave_pol, ref_pol)
        self.model.clear_actor_grad()
        loss.backward()
        if self.clip_actor_gradient:
            stats['grad_norm_actor'] = nn.utils.clip_grad_norm_(
                                            self.model.get_actor_params(), 
                                            self.actor_gradient_clip_value).item()
        self.actor_optim.step()
        return stats

    def _value_loss(self, obs, returns):
        """
        Computes the loss with current data. also returns a dictionary of statistics
        which includes value loss and explained variance
        return: surreal.utils.pytorch.GPUVariable, dict
        Args:
            obs: batch of observations in form of (batch_size, obs_dim)
            returns: batch of N-step return estimate (batch_size,)
        Returns:
            loss: Variable for loss
            stats: dictionary of recorded statistics
        """
        values = self.model.forward_critic(obs, self.cells)
        if len(values.size()) == 3: values = values.squeeze(2)
        explained_var = 1 - torch.var(returns - values) / torch.var(returns)
        loss = (values - returns).pow(2).mean()

        stats = {
            '_val_loss': loss.item(),
            '_val_explained_var': explained_var.item()
        }
        return loss, stats

    def _value_update(self, obs, returns):
        """
        Method that makes baseline function updates. calls _value_loss method
        Note:  self.clip_actor_gradient determines whether gradient is clipped
        return: dictionary of statistics to be sent to tensorplex server
        Args:
            obs: batch of observations in form of (batch_size, obs_dim)
            returns: batch of N-step return estimate (batch_size,)
        Returns:
            stats: dictionary of recorded statistics
        """
        loss, stats = self._value_loss(obs, returns)
        self.model.clear_critic_grad()
        loss.backward()
        if self.clip_critic_gradient:
            stats['grad_norm_critic'] = nn.utils.clip_grad_norm_(
                                                self.model.get_critic_params(), 
                                                self.critic_gradient_clip_value).item()
        self.critic_optim.step()
        return stats

    def _gae_and_return(self, obs, obs_next, rewards, dones):
        '''
        computes generalized advantage estimate and corresponding N-step return. 
        Details of algorithm can be found here: https://arxiv.org/pdf/1506.02438.pdf
        Args: 
            obs: batch of observations (batch_size, N-step , obs_dim)
            obs_next: batch of next observations (batch_size, 1 , obs_dim)
            actions: batch of actions (batch_size, N-step , act_dim)
            rewards: batch of rewards (batch_size, N-step)
            dones: batch of termination flags (batch_size, N-step)
        Returns:
            obs: batch of observation (batch_size, obs_dim)
            actions: batch of action (batch_size, act_dim)
            advantage: batch of advantages (batch_size, 1)
            returns: batch of returns (batch_size, 1)
        '''
        with tx.device_scope(self.gpu_option):
            index_set = torch.tensor(range(self.n_step), dtype=torch.float32)
            gamma = torch.pow(self.gamma, index_set)
            lam = torch.pow(self.lam, index_set)

            obs_concat_var = {}
            for mod in obs.keys():
                obs_concat_var[mod] = {}
                for k in obs[mod].keys():
                    obs_concat_var[mod][k] = (torch.cat([obs[mod][k], obs_next[mod][k]], dim=1))
                    if not self.if_rnn_policy:
                        obs_shape = obs_concat_var[mod][k].size()
                        obs_concat_var[mod][k] = obs_concat_var[mod][k].view(-1, *obs_shape[2:])

            values = self.model.forward_critic(obs_concat_var, self.cells) 
            values = values.view(self.batch_size, self.n_step + 1)
            values[:, 1:] *= 1 - dones

            if self.if_rnn_policy:
                tds = rewards + self.gamma * values[:, 1:] - values[:, :-1]
                eff_len = self.n_step - self.horizon + 1
                gamma = gamma[:self.horizon]
                lam = lam[:self.horizon]

                returns = torch.zeros(self.batch_size, eff_len)
                advs = torch.zeros(self.batch_size, eff_len)
                for step in range(eff_len):
                    returns[:, step] = torch.sum(gamma * rewards[:, step:step + self.horizon], 1) + \
                                       values[:, step + self.horizon] * (self.gamma ** self.horizon)
                    advs[:, step] = torch.sum(tds[:, step:step + self.horizon] * gamma * lam, 1)

                if self.norm_adv:
                    std = advs.std()
                    mean = advs.mean()
                    advs = (advs - mean) / max(std, 1e-4)
                return advs, returns

            else:
                returns = torch.sum(gamma * rewards, 1) + values[:, -1] * (self.gamma ** self.n_step)
                tds = rewards + self.gamma * values[:, 1:] - values[:, :-1] 
                gae = torch.sum(tds * gamma * lam, 1)

                if self.norm_adv:
                    std = gae.std()
                    mean = gae.mean()
                    gae = (gae - mean) / max(std, 1e-4)

                return gae.view(-1, 1), returns.view(-1, 1)

    def _preprocess_batch_ppo(self, batch):
        '''
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
        with tx.device_scope(self.gpu_option):

            obs, actions, rewards, obs_next, done, persistent_infos, onetime_infos = (
                batch['obs'],
                batch['actions'],
                batch['rewards'],
                batch['obs_next'],
                batch['dones'],
                batch['persistent_infos'],
                batch['onetime_infos'],
            )

            for modality in obs:
                for key in obs[modality]:
                    obs[modality][key] = (torch.tensor(obs[modality][key], dtype=torch.float32)).detach()
                    obs_next[modality][key] = (torch.tensor(obs_next[modality][key], dtype=torch.float32)).detach()

            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32) * self.reward_scale
            if self.use_r_filter:
                normed_reward = self.reward_filter.forward(rewards)
                self.reward_filter.update(rewards)
                rewards = normed_reward

            done = torch.tensor(done, dtype=torch.float32)

            if persistent_infos is not None:
                for i in range(len(persistent_infos)):
                    persistent_infos[i] = torch.tensor(persistent_infos[i], dtype=torch.float32).detach()
            if onetime_infos is not None:
                for i in range(len(onetime_infos)):
                    onetime_infos[i] = torch.tensor(onetime_infos[i], dtype=torch.float32).detach()

            (
                batch['obs'],
                batch['actions'],
                batch['rewards'],
                batch['obs_next'],
                batch['dones'],
                batch['persistent_infos'],
                batch['onetime_infos'],
            ) = (
                obs,
                actions,
                rewards,
                obs_next,
                done,
                persistent_infos,
                onetime_infos
            )
            return batch


    def _optimize(self, obs, actions, rewards, obs_next, persistent_infos, onetime_infos, dones):
        '''
            main method for optimization that calls _adapt/clip_update and 
            _value_update epoch_policy and epoch_baseline times respectively
            return: dictionary of tracted statistics
            Args:
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
        # convert everything to float tensor: 
        with tx.device_scope(self.gpu_option):
            pds = persistent_infos[-1]

            if self.if_rnn_policy:
                h = (onetime_infos[0].transpose(0, 1).contiguous()).detach()
                c = (onetime_infos[1].transpose(0, 1).contiguous()).detach()
                self.cells = (h, c)

            advantages, returns = self._gae_and_return(obs, 
                                                       obs_next,  
                                                       rewards, 
                                                       dones)
            advantages = advantages.detach()
            returns    = returns.detach()

            if self.if_rnn_policy:
                h = self.cells[0].detach()
                c = self.cells[1].detach()
                self.cells = (h, c)
                eff_len = self.n_step - self.horizon + 1
                behave_pol = pds[:, :eff_len, :].contiguous().detach()
                actions_iter = actions[:, :eff_len, :].contiguous().detach()
            else:
                behave_pol = pds[:, 0, :].contiguous().detach()
                actions_iter = actions[:, 0, :].contiguous().detach()

            obs_iter = {}
            for mod in obs.keys():
                obs_iter[mod] = {}
                for k in obs[mod].keys():
                    if self.if_rnn_policy:
                        obs_iter[mod][k] = obs[mod][k][:, :self.n_step - self.horizon + 1, :].contiguous().detach()
                    else: 
                        obs_iter[mod][k] = obs[mod][k][:, 0, :].contiguous().detach()

            ref_pol = self.ref_target_model.forward_actor(obs_iter, self.cells).detach()

            for ep in range(self.epoch_policy):
                if self.ppo_mode == 'clip':
                    stats =  self._clip_update(obs_iter, 
                                               actions_iter, 
                                               advantages, 
                                               behave_pol)
                else: 
                    stats = self._adapt_update(obs_iter, 
                                               actions_iter, 
                                               advantages, 
                                               behave_pol, 
                                               ref_pol)
                curr_pol = self.model.forward_actor(obs_iter, self.cells).detach()
                kl = self.pd.kl(ref_pol, curr_pol).mean()
                stats['_pol_kl'] = kl.item()
                if kl.item() > self.kl_target * 4: 
                    break

            self.kl_record.append(stats['_pol_kl'])

            for _ in range(self.epoch_baseline):
                baseline_stats = self._value_update(obs_iter, returns)

            # Collecting metrics and updating tensorplex
            for k in baseline_stats:
                stats[k] = baseline_stats[k]

            behave_likelihood = self.pd.likelihood(actions_iter, behave_pol)
            curr_likelihood   = self.pd.likelihood(actions_iter, curr_pol)

            stats['_avg_return_targ'] = returns.mean().item()
            stats['_avg_log_sig'] = self.model.actor.log_var.mean().item()
            stats['_avg_behave_likelihood'] = behave_likelihood.mean().item()
            stats['_avg_is_weight'] = (curr_likelihood / (behave_likelihood + 1e-4)).mean().item()
            stats['_ref_behave_diff'] = self.pd.kl(ref_pol, behave_pol).mean().item()
            stats['_lr'] = self.actor_lr_scheduler.get_lr()[0]

            if self.use_z_filter:
                self.model.z_update(obs_iter)
                stats['obs_running_mean'] = np.mean(self.model.z_filter.running_mean())
                stats['obs_running_square'] =  np.mean(self.model.z_filter.running_square())
                stats['obs_running_std'] = np.mean(self.model.z_filter.running_std())
            if self.use_r_filter:
                stats['reward_mean'] = self.reward_filter.reward_mean()

            return stats

    def learn(self, batch):
        '''
            main method for learning, calls _optimize. Also sends update stats 
            to Tensorplex
            Args:
                batch: pre-aggregated list of experiences rolled out by the agent
        '''
        self.current_iteration += 1
        batch = self._preprocess_batch_ppo(batch)
        tensorplex_update_dict = self._optimize(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.obs_next,
            batch.persistent_infos,
            batch.onetime_infos,
            batch.dones,
        )
        self.periodic_checkpoint(
            global_steps=self.current_iteration,
            score=None,
        )

        self.tensorplex.add_scalars(tensorplex_update_dict, self.global_step)
        self.exp_counter += self.batch_size
        self.global_step += 1

    def module_dict(self):
        '''
        returns the corresponding parameters
        '''
        return {
            'ppo': self.model,
        }

    def publish_parameter(self, iteration, message=''):
        """
        Learner publishes latest parameters to the parameter server only when 
        accumulated enough experiences specified by 
            learner_config.algo.network.update_target.interval
        Note: this overrides the base class publish_parameter method
        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        if  self.exp_counter >= self.learner_config.parameter_publish.exp_interval:
            self._ps_publisher.publish(iteration, message=message)
            self._post_publish()  

    def _post_publish(self):
        '''
            function that manages metrics and behavior after parameter release
            Actions include: 
                adjusts adaptive threshold for KL penalty for 'adapt' PPO 
                adjusts adaptive prob ratio clip rate for 'clip' PPO
                clears KL-Divergence record
                clears experience counter after parameter release
                steps actor and critic learning rate scheduler
        '''
        final_kl = np.mean(self.kl_record)
        if self.ppo_mode == 'clip': # adapts clip ratios
            if final_kl > self.kl_target * self.clip_adjust_threshold[1]:
                if self.clip_lower < self.clip_epsilon:
                    self.clip_epsilon = self.clip_epsilon / self.learner_config.algo.clip_consts.scale_constant
            elif final_kl < self.kl_target * self.clip_adjust_threshold[0]:
                if self.clip_upper > self.clip_epsilon:
                    self.clip_epsilon = self.clip_epsilon * self.learner_config.algo.clip_consts.scale_constant
        else: # adapt KL divergence penalty before returning the statistics 
            if final_kl > self.kl_target * self.beta_adjust_threshold[1]:
                if self.beta_upper > self.beta:
                    self.beta = self.beta * self.learner_config.algo.adapt_consts.scale_constant
            elif final_kl < self.kl_target * self.beta_adjust_threshold[0]:
                if self.beta_lower < self.beta:
                    self.beta = self.beta / self.learner_config.algo.adapt_consts.scale_constant
        self.ref_target_model.update_target_params(self.model)
        self.kl_record = []
        self.exp_counter = 0
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

    def checkpoint_attributes(self):
        '''
            outlines attributes to be checkpointed
        '''
        return [
            'model',
            'ref_target_model',
            'actor_lr_scheduler',
            'critic_lr_scheduler',
            'current_iteration',
        ]

    def _prefetcher_preprocess(self, batch):
        batch = self.aggregator.aggregate(batch)
        return batch
