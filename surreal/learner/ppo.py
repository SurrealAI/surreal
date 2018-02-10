import torch
import torch.nn as nn
import numpy as np
from .base import Learner
from .aggregator import MultistepWithBehaviorPolicyAggregator 
from surreal.model.ppo_net import PPOModel, DiagGauss
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG, ConfigError
from surreal.utils.pytorch import GpuVariable as Variable
import surreal.utils as U

import pdb

class PPOLearner(Learner):

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)

        # RL general parameters
        self.gamma = self.learner_config.algo.gamma
        self.n_step = self.learner_config.algo.n_step
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.norm_adv = self.learner_config.algo.norm_adv
        self.batch_size = self.learner_config.algo.batch_size

        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_dim = self.env_config.obs_spec.dim[0]
        self.model = PPOModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
        )

        # PPO parameters
        self.method = self.learner_config.algo.method
        self.trace_cutoff = self.learner_config.algo.trace_cutoff
        self.is_weight_thresh = self.learner_config.algo.is_weight_thresh
        self.is_weight_eps = self.learner_config.algo.is_weight_eps
        self.lr_policy = self.learner_config.algo.lr_policy
        self.lr_baseline = self.learner_config.algo.lr_baseline
        self.epoch_policy = self.learner_config.algo.epoch_policy
        self.epoch_baseline = self.learner_config.algo.epoch_baseline
        self.kl_targ = self.learner_config.algo.kl_targ
        self.kl_cutoff_coeff = self.learner_config.algo.kl_cutoff_coeff
        self.clip_epsilon_init = self.learner_config.algo.clip_epsilon_init
        self.beta_init = self.learner_config.algo.beta_init
        self.clip_range = self.learner_config.algo.clip_range
        self.adj_thres = self.learner_config.algo.adj_thres
        self.beta_range = self.learner_config.algo.beta_range

        if self.method == 'adapt':
            self.beta = self.beta_init
            self.eta = self.kl_cutoff_coeff
            self.beta_upper = self.beta_range[1]
            self.beta_lower = self.beta_range[0]
            self.beta_adj_thres = self.adj_thres
        else: # method == 'clip'
            self.clip_epsilon = self.clip_epsilon_init
            self.clip_adj_thres = self.adj_thres
            self.clip_upper = self.clip_range[1]
            self.clip_lower = self.clip_range[0]

        # Learning parameters
        self.clip_actor_gradient = self.learner_config.algo.clip_actor_gradient
        if self.clip_actor_gradient:
            self.actor_gradient_clip_value = self.learner_config.algo.actor_gradient_clip_value

        self.clip_critic_gradient = self.learner_config.algo.clip_critic_gradient
        if self.clip_critic_gradient:
            self.critic_gradient_clip_value = self.learner_config.algo.critic_gradient_clip_value

        self.critic_optim = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=self.lr_baseline
        )
        self.actor_optim = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=self.lr_policy
        )

        # Experience Aggregator
        self.aggregator = MultistepWithBehaviorPolicyAggregator(self.env_config.obs_spec, self.env_config.action_spec)
        
        # probability distribution. Gaussian only for now
        self.pd = DiagGauss(self.action_dim)


    def _advantage_and_return(self, obs, actions, rewards, obs_next, done, num_steps):
        n_samples = obs.size()[0]
        gamma = torch.ones(n_samples, 1) * self.gamma

        values = self.model.critic(Variable(obs)).detach().data
        next_values = self.model.critic(Variable(obs_next)).detach().data
        returns = rewards + next_values * (1 - done) * torch.pow(gamma, num_steps) # value bootstrap
        adv = returns - values

        if self.norm_adv:
            std = adv.std()
            mean = adv.mean()
            adv = (adv - mean) / std

        return adv, returns


    def _clip_loss(self, obs, actions, advantages, fixed_dist, fixed_prob): 
        new_prob = self.model.actor(obs)
        new_p = self.pd.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr = -prob_ratio * advantages
        cliped_surr = -cliped_ratio * advantages
        clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean()

        stats = {
            "surr_loss": surr.mean().data[0], 
            "clip_surr_loss": clip_loss.data[0], 
            "entropy": self.pd.entropy(new_prob).data.mean(),
            'clip_epsilon': self.clip_epsilon
        }

        return clip_loss, stats


    def _clip_update_iter(self, obs, actions, advantages, behave_pol):
        behave_pol = self.model.actor(obs).detach()
        fixed_prob = self.pd.likelihood(actions, behave_pol).detach() + self.is_weight_eps # variance reduction step

        num_batches = obs.size()[0] // self.batch_size + 1
        for epoch in range(self.epoch_policy):
            sortinds = np.random.permutation(obs.size()[0])
            sortinds = Variable(torch.from_numpy(sortinds).long())
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                if start >= obs.size()[0]: break

                obs_batch = obs.index_select(0, sortinds[start:end])
                act_batch = actions.index_select(0, sortinds[start:end])
                adv_batch = advantages.index_select(0, sortinds[start:end])
                prb_batch = behave_pol.index_select(0, sortinds[start:end])
                fix_batch = fixed_prob.index_select(0, sortinds[start:end])

                loss, _ = self._clip_loss(obs_batch, act_batch, adv_batch, prb_batch, fix_batch)
                self.model.actor.zero_grad()
                loss.backward()
                self.actor_optim.step()

            prob = self.model.actor(obs)
            kl = self.pd.kl(old_prob, prob).mean()
            if kl.data[0] > 4 * self.kl_targ:
                break

        if kl.data[0] > self.kl_targ * self.clip_adj_thres[1]:
            if self.clip_lower < self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon / 1.2
        elif kl.data[0] < self.kl_targ * self.clip_adj_thres[0]:
            if self.clip_upper > self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon * 1.2
        
        _, stats = self._clip_loss(obs, actions, advantages, behave_pol, fixed_prob)
        return stats


    def _clip_update_full(self, obs, actions, advantages, behave_pol):
        behave_pol = self.model.actor(obs).detach()
        fixed_prob = self.pd.likelihood(actions, behave_pol).detach() + self.is_weight_eps # variance reduction step
        for epoch in range(self.epoch_policy):

            loss, stats = self._clip_loss(obs, actions, advantages, behave_pol, fixed_prob)
            self.model.actor.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.model.actor.parameters(), self.actor_gradient_clip_value)
            self.actor_optim.step()

            prob = self.model.actor(obs)
            kl = self.pd.kl(behave_pol, prob).mean()
            stats['pol_kl'] = kl.data[0]
            if kl.data[0] > 4 * self.kl_targ:
                break

        if kl.data[0] > self.kl_targ * self.clip_adj_thres[1]:
            if self.clip_lower < self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon / 1.2
        elif kl.data[0] < self.kl_targ * self.clip_adj_thres[0]:
            if self.clip_upper > self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon * 1.2
        
        return stats


    def _adapt_loss(self, obs, actions, advantages, old_pol):
        prob = self.model.actor(obs)
        logp = self.pd.loglikelihood(actions, prob)
        logp_old = self.pd.loglikelihood(actions, old_prob)
        kl = self.pd.kl(old_prob, prob).mean()
        surr = -(advantages * (logp - logp_old).exp()).mean()
        loss = surr + self.beta * kl

        if kl.data[0] - 2.0 * self.kl_targ > 0:
            loss += self.eta * (kl - 2.0 * self.kl_targ).pow(2)

        stats = {
            'kl_loss_adapt': loss.data[0], 
            'surr_loss': surr.data[0], 
            'pol_kl': kl.data[0], 
            'entropy': entropy.data[0],
            'beta': self.beta
        }

        return loss, stats

    def _adapt_update_full(self, obs, actions, advantages, pds):
        old_prob = self.model.actor(obs).detach()

        for epoch in range(self.epoch_policy):

            loss, _ = self._adapt_loss(obs, actions, advantages, old_prob)
            self.model.actor.zero_grad()
            loss.backward()
            self.actor_optim.step()

            prob = self.model.actor(obs)
            kl = self.pd.kl(old_prob, prob).mean()
            stats['pol_kl'] = kl.data[0]
            if kl.data[0] > self.kl_targ * 4:
                break

        if kl.data[0] > self.kl_targ * self.beta_adj_thres[1]:
            if self.beta_upper > self.beta:
                self.beta = self.beta * 1.5
        elif kl.data[0] < self.kl_targ * self.beta_adj_thres[0]:
            if self.beta_lower < self.beta:
                self.beta = self.beta / 1.5

        return stats


    def _adapt_update_iter(self, obs, actions, advantages, pds):
        old_prob = self.model.actor(obs).detach()

        num_batches = obs.size()[0] // self.batch_size + 1
        for epoch in range(self.epoch_policy):
            sortinds = np.random.permutation(obs.size()[0])
            sortinds = torch.autograd.Variable(turn_into_cuda(torch.from_numpy(sortinds).long()))
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                if start >= obs.size()[0]: break

                obs_batch = obs.index_select(0, sortinds[start:end])
                act_batch = actions.index_select(0, sortinds[start:end])
                adv_batch = advantages.index_select(0, sortinds[start:end])
                prb_batch = old_prob.index_select(0, sortinds[start:end])

                loss, _ = self._adapt_loss(obs_batch, act_batch, adv_batch, prb_batch)
                self.model.actor.zero_grad()
                loss.backward()
                self.actor_optim.step()

            prob = self.model.actor(obs)
            kl = self.pd.kl(old_prob, prob).mean()
            if kl.data[0] > self.kl_targ * 4:
                break

        if kl.data[0] > self.kl_targ * self.beta_adj_thres[1]:
            if self.beta_upper > self.beta:
                self.beta = self.beta * 1.5
        elif kl.data[0] < self.kl_targ * self.beta_adj_thres[0]:
            if self.beta_lower < self.beta:
                self.beta = self.beta / 1.5

        _, stats = self._adapt_loss(obs, actions, advantages, old_prob)
        return stats


    def _value_loss(self, obs, returns):
        values = self.model.critic(obs)
        explained_var = 1 - torch.var(returns - values) / torch.var(returns)
        loss = (values - returns).pow(2).mean()

        stats = {
            'val_loss': loss.data[0],
            'val_explained_var': explained_var.data[0]
        }
        return loss, stats


    def _value_update_iter(self, obs, returns):
        num_batches = obs.size()[0] // self.batch_size + 1
        for epoch in range(self.epoch_baseline):
            sortinds = np.random.permutation(obs.size()[0])
            sortinds = Variable(torch.from_numpy(sortinds).long())
            for j in range(num_batches):
                start = j * self.batch_size
                end = (j + 1) * self.batch_size
                if start >= obs.size()[0]: break

                obs_batch = obs.index_select(0, sortinds[start:end])
                ret_batch = returns.index_select(0, sortinds[start:end])

                loss, _ = self._value_loss(obs_batch, ret_batch)
                self.model.critic.zero_grad()
                loss.backward()
                self.critic_optim.step()

        _, stats = self._value_loss(obs, returns)
        return stats


    def _value_update_full(self, obs, returns):
        for epoch in range(self.epoch_baseline):
            loss, stats = self._value_loss(obs, returns)
            self.model.critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.model.critic.parameters(), self.critic_gradient_clip_value)
            self.critic_optim.step()

        return stats


    def _V_trace_compute_target(self, obs, obs_next, actions, rewards, pds, dones):
        batch_size =self.learner_config.replay.batch_size
        obs_flat = obs.view(batch_size * self.n_step, -1 ) 
        actions_flat = actions.view(batch_size * self.n_step, -1)

        # getting the importance sampling weights
        curr_pd   = self.model.actor(Variable(obs_flat))
        curr_prob = self.pd.likelihood(Variable(actions_flat), curr_pd).data # tensor

        behave_pd   = Variable(pds.view(batch_size * self.n_step, -1))
        behave_prob = self.pd.likelihood(Variable(actions_flat), behave_pd).data
        behave_prob = torch.clamp(behave_prob, min=self.is_weight_eps) # hard clamp, not variance reduction step

        is_weight = curr_prob / behave_prob
        is_weight_trunc = torch.clamp(is_weight, max=self.is_weight_thresh)
        trace_c         = torch.clamp(is_weight, max=self.trace_cutoff) # both in shape (batch * n, 1)
    
        is_weight_trunc = is_weight_trunc.view(batch_size, self.n_step)
        log_trace_c     = torch.log(trace_c.view(batch_size, self.n_step)) # for numeric stability
        log_trace_c = log_trace_c.cumsum(1)
        trace_c = log_trace_c.exp()

        # value estimate
        obs_concat_flat = torch.cat((obs, obs_next), dim = 1).view(batch_size * (self.n_step + 1), -1)
        values = self.model.critic(Variable(obs_concat_flat)).data # (batch * (n+1), 1)
        values = values.view(batch_size, self.n_step + 1)
        values[:, :self.n_step] *= 1 - dones
        values[:, self.n_step]  *= 1 - dones[:, -1]

        # computing trace
        delta = is_weight_trunc * (rewards + self.gamma * values[:, 1:]) - values[:, :-1] # (batch, n)
        gamma = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step)))
        #inv_idx = torch.arange(log_trace_c.size(1)-1, -1, -1).long() 
        #inv_log_trace = log_trace_c.index_select(1, inv_idx)
        #inv_log_trace = inv_log_trace.cumsum(dim = 1)
        #inv_trace = torch.exp(inv_log_trace) # (batch, n)
        #trace = inv_trace.index_select(1, inv_idx)

        # More efficient, more biased. run with stride ~= n_step
        adv  = trace_c * delta
        for step in range(self.n_step):
            gamma = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step - step))) # (n -1,)
            adv[:, step] = torch.sum(adv[:, step:] * gamma, dim=1)
        v_trace_targ = adv + values[:, :-1]
        adv = adv.view(-1 , 1)
        v_trace_targ = v_trace_targ.view(-1, 1)
        pds_flat = pds.view(batch_size * self.n_step, -1)
        # Less Efficnet, less biased. Run with stride = 1
        '''
        gamma = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step)))
        adv = torch.sum(trace_c * gamma * delta, dim = 1)
        v_trace_targ = adv + values[:, 0]
        obs_flat = obs[:, 0, :].squeeze(1)
        action_flat = actions[:, 0, :].squeeze(1) 
        pds_flat = pds[:, 0, :].squeeze(1)
        '''

        if self.norm_adv:
            mean = adv.mean()
            std  = adv.std()
            adv  = (adv - mean)/std  

        return obs_flat, actions_flat, adv, v_trace_targ, pds_flat    


    def _optimize(self, obs, actions, rewards, obs_next, pds, dones):
        obs, actions, advantages, v_trace_targ, pds = self._V_trace_compute_target(obs, obs_next, actions, 
                                                                                   rewards, pds, dones)
        obs = Variable(obs)
        actions = Variable(actions)
        advantages = Variable(advantages)
        v_trace_targ = Variable(v_trace_targ)
        pds = Variable(pds)


        if self.method == 'clip': 
            stats = self._clip_update_full(obs, actions, advantages, pds)
        else:
            stats = self._adapt_update_full(obs, actions, advantages, pds)
        baseline_stats = self._value_update_full(obs, v_trace_targ)

        # updating tensorplex
        for k in baseline_stats:
            stats[k] = baseline_stats[k]
        stats['avg_vtrace_targ'] = v_trace_targ.mean().data[0]
        stats['avg_log_sig'] = self.model.actor.log_var.mean().data[0]
        stats['avg_behave_prob'] = self.pd.likelihood(actions, pds).mean().data
        new_pol_pd = self.model.actor(obs)
        new_likelihood = self.pd.likelihood(actions, new_pol_pd)
        stats['avg_IS_weight'] = (new_likelihood/torch.clamp(self.pd.likelihood(actions, pds), min = 1e-5)).mean().data

        if self.use_z_filter:
            stats['observation_0_running_mean'] = self.model.z_filter.running_mean()[0]
            stats['observation_0_running_square'] =  self.model.z_filter.running_square()[0]
            stats['observation_0_running_std'] = self.model.z_filter.running_std()[0]

        self.update_tensorplex(stats)

    def learn(self, batch):
        batch = self.aggregator.aggregate(batch)
        self._optimize(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.next_obs,
            batch.pds, 
            batch.dones,
        )

    def module_dict(self):
        return {
            'ppo': self.model,
        }

'''
    off-policy bias corrected PPO: -> custom V-trace aggregator 
        1) V-trace integration:
            * advantage estimate with V-trace
            * value update with V-trace
            * epsilon variance reduction in IS weights
        2) Trajectory: using partial trajectories instead full trajectories
            * Caveat: introduces bias
            * CAN use full trajectory but it also reduces learning frequency (trade off)

        Implementation details (compared to regular PPO):
            - partial trajectoriy (N-step) | bias  ^
            - sample size  v
            - stride match N-step (or truncating)
            - aggregator: multistep aggregator instead of NstepReturnAggregator


        TODO:
            - √ custom sender
            - √ high bias, efficient implementation
            - low bias, inefficient implementation
            - agent side sleeping?
            - log_sig smaller learning rate
'''
