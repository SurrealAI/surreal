import torch
import torch.nn as nn
import numpy as np
from .base import Learner
from .aggregator import MultistepWithBehaviorPolicyAggregator 
from surreal.model.ppo_net import PPOModel, DiagGauss
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG, ConfigError
from surreal.utils.pytorch import GpuVariable as Variable
import surreal.utils as U

import time

class PPOLearner(Learner):

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)

        # GPU setting
        num_gpus = session_config.learner.num_gpus
        self.gpu_id = list(range(num_gpus))
        self.use_cuda = len(self.gpu_id) > 0

        if not self.use_cuda:
            self.log.info('Using CPU')
        else:
            self.log.info('Using GPU: {}'.format(self.gpu_id))

        # RL general parameters
        self.gamma = self.learner_config.algo.gamma
        self.lam   = self.learner_config.algo.lam
        self.n_step = self.learner_config.algo.n_step
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.norm_adv = self.learner_config.algo.norm_adv
        self.batch_size = self.learner_config.algo.batch_size

        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_dim = self.env_config.obs_spec.dim[0]
        self.init_log_sig = self.learner_config.algo.init_log_sig
        self.model = PPOModel(
            init_log_sig=self.init_log_sig,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
            use_cuda = self.use_cuda, 
        )
        self.ref_target_model = PPOModel(
            init_log_sig=self.init_log_sig,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
            use_cuda = self.use_cuda, 
        )
        self.ref_target_model.update_target_params(self.model)

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

        self.exp_counter = 0
        self.kl_record = []

        with U.torch_gpu_scope(self.gpu_id):

            # Learning parameters
            self.clip_actor_gradient = self.learner_config.algo.clip_actor_gradient
            self.actor_gradient_clip_value = self.learner_config.algo.actor_gradient_clip_value
            self.clip_critic_gradient = self.learner_config.algo.clip_critic_gradient
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

    def _clip_loss(self, obs, actions, advantages, fixed_dist, fixed_prob): 
        new_prob = self.model.forward_actor(obs)
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


    def _clip_update_full(self, obs, actions, advantages, behave_pol, ref_pol):
        behave_pol = self.model.forward_actor(obs).detach()
        fixed_prob = self.pd.likelihood(actions, behave_pol).detach() + self.is_weight_eps # variance reduction step
        for epoch in range(self.epoch_policy):

            loss, stats = self._clip_loss(obs, actions, advantages, behave_pol, fixed_prob)
            self.model.actor.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.model.actor.parameters(), self.actor_gradient_clip_value)
            self.actor_optim.step()

            prob = self.model.forward_actor(obs)
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


    def _adapt_loss(self, obs, actions, advantages, behave_pol, ref_pol):
        learn_pol = self.model.forward_actor(obs)
        prob_behave = self.pd.likelihood(actions, behave_pol)
        prob_learn  = self.pd.likelihood(actions, learn_pol)
        kl = self.pd.kl(ref_pol, learn_pol).mean()
        surr = -(advantages * torch.clamp(prob_learn/prob_behave, max=self.is_weight_thresh)).mean()
        loss = surr + self.beta * kl
        entropy = self.pd.entropy(learn_pol).mean()

        if kl.data[0] - 2.0 * self.kl_targ > 0:
            loss += self.eta * (kl - 2.0 * self.kl_targ).pow(2)

        stats = {
            '_kl_loss_adapt': loss.data[0], 
            '_surr_loss': surr.data[0], 
            '_pol_kl': kl.data[0], 
            '_entropy': entropy.data[0],
            '_beta': self.beta
        }

        return loss, stats


    def _adapt_update_sim(self, obs, actions, advantages, behave_pol, ref_pol):
        loss, stats = self._adapt_loss(obs, actions, advantages, behave_pol, ref_pol)
        self.model.actor.zero_grad()
        loss.backward()
        if self.clip_actor_gradient:
            nn.utils.clip_grad_norm(self.model.actor.parameters(), self.actor_gradient_clip_value)
        self.actor_optim.step()
        return stats

    def _adapt_update_full(self, obs, actions, advantages, behave_pol, ref_pol):

        for epoch in range(self.epoch_policy):
            loss, stats = self._adapt_loss(obs, actions, advantages, behave_pol, ref_pol)
            self.model.actor.zero_grad()
            loss.backward()
            if self.clip_actor_gradient:
                nn.utils.clip_grad_norm(self.model.actor.parameters(), self.actor_gradient_clip_value)
            self.actor_optim.step()

            curr_pol = self.model.forward_actor(obs)
            kl = self.pd.kl(ref_pol, curr_pol).mean()
            stats['_pol_kl'] = kl.data[0]
            if kl.data[0] > self.kl_targ * 4:
                break

        if kl.data[0] > self.kl_targ * self.beta_adj_thres[1]:
            if self.beta_upper > self.beta:
                self.beta = self.beta * 1.5
        elif kl.data[0] < self.kl_targ * self.beta_adj_thres[0]:
            if self.beta_lower < self.beta:
                self.beta = self.beta / 1.5

        return stats


    def _value_loss(self, obs, returns):
        values = self.model.forward_critic(obs)
        explained_var = 1 - torch.var(returns - values) / torch.var(returns)
        loss = (values - returns).pow(2).mean()

        stats = {
            '_val_loss': loss.data[0],
            '_val_explained_var': explained_var.data[0]
        }
        return loss, stats

    def _value_update_full(self, obs, returns):
        for epoch in range(self.epoch_baseline):
            loss, stats = self._value_loss(obs, returns)
            self.model.critic.zero_grad()
            loss.backward()
            if self.clip_critic_gradient:
                nn.utils.clip_grad_norm(self.model.critic.parameters(), self.critic_gradient_clip_value)
            self.critic_optim.step()

        return stats


    def _value_update_sim(self, obs, returns):
        loss, stats = self._value_loss(obs, returns)
        self.model.critic.zero_grad()
        loss.backward()
        if self.clip_critic_gradient:
            nn.utils.clip_grad_norm(self.model.critic.parameters(), self.critic_gradient_clip_value)
        self.critic_optim.step()

        return stats

    def _gae_and_return(self, obs, obs_next, actions, rewards, pds, dones):

        with U.torch_gpu_scope(self.gpu_id):
            gamma = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step)))
            lam = torch.pow(self.lam, U.to_float_tensor(range(self.n_step)))
            batch_size =self.learner_config.replay.batch_size 
                
            if self.use_cuda: 
                rewards = rewards.cuda()
                dones = dones.cuda()
                gamma = gamma.cuda()
                lam   = lam.cuda()

            obs_concat = torch.cat([obs, obs_next], dim=1).view(batch_size * (self.n_step + 1), -1)
            values = self.model.forward_critic(Variable(obs_concat)).data # (batch * (n+1), 1)
            values = values.view(batch_size, self.n_step + 1)
            values[:, 1:] *= 1 - dones

            returns = torch.sum(gamma * rewards, 1) + values[:, -1] * (self.gamma ** self.n_step)

            tds = rewards + self.gamma * values[:, 1:] - values[:, :-1] 
            gae = torch.sum(tds * gamma * lam, 1)

            if self.norm_adv:
                std = gae.std()
                mean = gae.mean()
                gae = (gae - mean) / std

            return obs[:, 0, :], actions[:, 0, :], gae.view(-1, 1), returns.view(-1, 1), pds[:, 0, :], 0

    def _advantage_and_return(self, obs, obs_next, actions, rewards, pds, dones):
        # assumes obs come in the shape of (batch_size, n_step, obs_dim)
        with U.torch_gpu_scope(self.gpu_id):
            gamma = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step)))
            if self.use_cuda: 
                rewards = rewards.cuda()
                dones = dones.cuda()
                gamma = gamma.cuda()

            done_0 = dones[:, 0].contiguous()
            done_n = dones[:, -1].contiguous()
            obs_flat = obs[:, 0, :] # (batch, obs_dim)
            actions_flat = actions[:, 0, :] # (batch, act_dim)
            pds_flat = pds[:, 0, :] # (batch, 2* act_dim)

            values   = self.model.forward_critic(Variable(obs_flat)).data * (1 - done_0.view(-1, 1))
            next_val = self.model.forward_critic(Variable(obs_next.squeeze(1))).data * (1 - done_n.view(-1, 1))

            returns = torch.sum(rewards * gamma, 1).view(-1, 1)
            returns = returns + next_val * (self.gamma ** self.n_step) # value bootstrap
            adv = returns - values

            if self.norm_adv:
                std = adv.std()
                mean = adv.mean()
                adv = (adv - mean) / std
                
            return obs_flat, actions_flat, adv, returns, pds_flat, 0

    def _V_trace_compute_target(self, obs, obs_next, actions, rewards, pds, dones):
        with U.torch_gpu_scope(self.gpu_id):

                if self.use_cuda: 
                    rewards = rewards.cuda()
                    dones = dones.cuda()

                batch_size =self.learner_config.replay.batch_size
                obs_flat = obs.view(batch_size * self.n_step, -1 ) 
                actions_flat = actions.view(batch_size * self.n_step, -1)

                # getting the importance sampling weights
                curr_pd   = self.model.forward_actor(Variable(obs_flat))
                curr_prob = self.pd.likelihood(Variable(actions_flat), curr_pd).data # tensor

                behave_pd   = Variable(pds.view(batch_size * self.n_step, -1))
                behave_prob = self.pd.likelihood(Variable(actions_flat), behave_pd).data
                behave_prob = torch.clamp(behave_prob, min=self.is_weight_eps) # hard clamp, not variance reduction step

                is_weight = curr_prob / behave_prob
                is_weight_trunc = torch.clamp(is_weight, max=self.is_weight_thresh)
                trace_c         = torch.clamp(is_weight, max=self.trace_cutoff) # both in shape (batch * n, 1)
            
                is_weight_trunc = is_weight_trunc.view(batch_size, self.n_step)
                log_trace_c     = torch.log(trace_c.view(batch_size, self.n_step)) # for numeric stability
                log_trace_c     = log_trace_c.cumsum(1)
                trace_c         = log_trace_c.exp()[:,:-1] # (batch_size, n-1)

                # value estimate
                obs_concat_flat = torch.cat((obs, obs_next), dim = 1).view(batch_size * (self.n_step + 1), -1)
                values = self.model.forward_critic(Variable(obs_concat_flat)).data # (batch * (n+1), 1)
                values = values.view(batch_size, self.n_step + 1)
                values[:, 1:] *= 1 - dones

                # computing trace
                tds = is_weight_trunc * (rewards + self.gamma * values[:, 1:] - values[:, :-1]) # (batch, n)
                tds[:, 1:] *= trace_c

                # More efficient, more biased. run with stride ~= n_step
                # v_trace_s_adv = tds
                # for step in range(self.n_step):
                #     gamma = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step - step))) # (n - i,)
                #     if self.use_cuda:
                #         gamma = gamma.cuda()
                #     v_trace_s_adv[:, step] = torch.sum(v_trace_s_adv[:, step:] * gamma, dim=1)

                # v_trace_s = v_trace_s_adv + values[:, :-1]
                # v_trace_s_adv = v_trace_s_adv.view(-1 , 1)
                # v_trace_s = v_trace_s.view(-1, 1)
                # pds_flat = pds.view(batch_size * self.n_step, -1)

                # Less Efficnet, less biased. Run with stride = 1                
                gamma_s_plus_1 = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step - 1))) # (n-1,)
                gamma_s        = torch.pow(self.gamma, U.to_float_tensor(range(self.n_step)))
                if self.use_cuda:
                    gamma_s = gamma_s.cuda()
                    gamma_s_plus_1 = gamma_s_plus_1.cuda()

                v_trace_s_plus_1 = values[:, 1] + torch.sum(tds[:, 1:] * gamma_s_plus_1, dim = 1) #(batch_size,)
                # v_trace_s_adv = torch.sum(tds * gamma_s, dim = 1)
                # v_trace_s     = v_trace_s_adv + values[:, 0]
                v_trace_s = values[:, 0] + tds[:, 0] + self.gamma * trace_c[:, 0] * (v_trace_s_plus_1 - values[:, 1]) # (batch_size,)
                v_trace_s_adv = rewards[:, 0] + self.gamma * v_trace_s_plus_1 - values[:, 0]  # (batch_size,)

                obs_flat = obs[:, 0, :] # (batch, obs_dim)
                actions_flat = actions[:, 0, :] # (batch, act_dim)
                pds_flat = pds[:, 0, :] # (batch, 2* act_dim)
                v_trace_s_adv = v_trace_s_adv.view(-1, 1)
                v_trace_s = v_trace_s.view(-1, 1)                
                

                if self.norm_adv:
                    mean = v_trace_s_adv.mean()
                    std  = v_trace_s_adv.std()
                    v_trace_s_adv  = (v_trace_s_adv - mean)/std  

                avg_trace = trace_c.mean()
                return obs_flat, actions_flat, v_trace_s_adv, v_trace_s, pds_flat, avg_trace

    def _optimize(self, obs, actions, rewards, obs_next, pds, dones):
        
        with U.torch_gpu_scope(self.gpu_id):
                obs_iter, actions_iter, advantages, v_trace_targ, behave_pol, avg_trace = self._gae_and_return(obs, obs_next, actions, rewards, pds, dones)
                obs_iter = Variable(obs_iter)
                actions_iter = Variable(actions_iter)
                advantages = Variable(advantages)
                v_trace_targ = Variable(v_trace_targ)
                behave_pol = Variable(behave_pol)

                # Target Ref GAE:
                # --------------------------------------------------------------------------------------------
                ref_pol = self.ref_target_model.forward_actor(obs_iter).detach()
                for _ in range(self.epoch_policy):
                    stats = self._adapt_update_sim(obs_iter, actions_iter, advantages, behave_pol, ref_pol)
                    curr_pol = self.model.forward_actor(obs_iter).detach()
                    kl = self.pd.kl(ref_pol, curr_pol).mean()
                    stats['_pol_kl'] = kl.data[0]
                    if kl.data[0] > self.kl_targ * 4: break
                self.kl_record.append(stats['_pol_kl'])

                for _ in range(self.epoch_baseline):
                    baseline_stats = self._value_update_sim(obs_iter, v_trace_targ)


                # Vtrace
                # --------------------------------------------------------------------------------------------
                # Simultaneous updates - updates actor and critic each for one step and computes the advantage again
                # uses ExpSenderWrapperMultiStepBehavePolicyMovingWindow exp_sender and MultistepWithBehaviorPolicyAggregator
                # done_training = False
                # ref_pol = self.model.forward_actor(Variable(obs[:, 0, :])).detach()
                # for _ in range(self.epoch_policy):
                #     obs_iter, actions_iter, advantages, v_trace_targ, behave_pol, avg_trace = self._V_trace_compute_target(obs, obs_next, actions, rewards, pds, dones)
                #     obs_iter = Variable(obs_iter)
                #     actions_iter= Variable(actions_iter)
                #     advantages = Variable(advantages)
                #     v_trace_targ = Variable(v_trace_targ)
                #     behave_pol = Variable(behave_pol)

                #     if self.method == 'clip':
                #         stats = self._clip_update_sim(obs_iter, actions_iter, advantages, behave_pol, ref_pol)
                #     else:
                #         stats = self._adapt_update_sim(obs_iter, actions_iter, advantages, behave_pol, ref_pol)
                        
                #     curr_pol = self.model.forward_actor(obs_iter).detach()
                #     kl = self.pd.kl(ref_pol, curr_pol).mean()
                #     stats['_pol_kl'] = kl.data[0]
                #     if kl.data[0] > self.kl_targ * 4:
                #         done_training = True

                #     baseline_stats = self._value_update_sim(obs_iter, v_trace_targ)
                #     if done_training: break

                # if self.method == 'clip':
                #     pass 
                # else: # adapt KL divergence penalty before returning the statistics 
                #     if stats['_pol_kl'] > self.kl_targ * self.beta_adj_thres[1]:
                #         if self.beta_upper > self.beta:
                #             self.beta = self.beta * 1.5
                #     elif stats['_pol_kl'] < self.kl_targ * self.beta_adj_thres[0]:
                #         if self.beta_lower < self.beta:
                #             self.beta = self.beta / 1.5



                # updating tensorplex
                for k in baseline_stats:
                    stats[k] = baseline_stats[k]

                new_pol = self.model.forward_actor(obs_iter).detach()
                new_likelihood = self.pd.likelihood(actions_iter, new_pol)
                ref_likelihood = self.pd.likelihood(actions_iter, ref_pol)
                behave_likelihood = self.pd.likelihood(actions_iter, behave_pol)

                stats['_avg_return_targ'] = v_trace_targ.mean().data[0]
                stats['_avg_log_sig'] = self.model.actor.log_var.mean().data[0]
                stats['_avg_behave_likelihood'] = behave_likelihood.mean().data[0]
                stats['_ref_behave_diff'] = self.pd.kl(ref_pol, behave_pol).mean().data[0]
                stats['_avg_IS_weight'] = (new_likelihood/torch.clamp(behave_likelihood, min = 1e-5)).mean().data[0]
                stats['_avg_trace'] = avg_trace

                if self.use_z_filter:
                    self.model.z_update(obs_iter)
                    stats['obs_running_mean'] = np.mean(self.model.z_filter.running_mean())
                    stats['obs_running_square'] =  np.mean(self.model.z_filter.running_square())
                    stats['obs_running_std'] = np.mean(self.model.z_filter.running_std())
                    self.ref_target_model.update_target_z_filter(self.model)

        return stats

    def learn(self, batch):
        batch = self.aggregator.aggregate(batch)
        tensorplex_update_dict = self._optimize(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.next_obs,
            batch.pds, 
            batch.dones,
        )
        self.tensorplex.add_scalars(tensorplex_update_dict)
        self.exp_counter += self.learner_config.replay.batch_size

    def module_dict(self):
        return {
            'ppo': self.model,
        }

    def post_publish(self):
        final_kl = np.mean(self.kl_record)
        if self.method == 'clip':
            pass 
        else: # adapt KL divergence penalty before returning the statistics 
            if final_kl > self.kl_targ * self.beta_adj_thres[1]:
                if self.beta_upper > self.beta:
                    self.beta = self.beta * 1.5
            elif final_kl < self.kl_targ * self.beta_adj_thres[0]:
                if self.beta_lower < self.beta:
                    self.beta = self.beta / 1.5
        self.ref_target_model.update_target_params(self.model)

    # overriding base class method to implement learner side count based publish
    def publish_parameter(self, iteration, message=''):
        """
        Learner publishes latest parameters to the parameter server only when accumulated
            enough experiences
        Note: this overrides the base class publish_parameter method

        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        if self.exp_counter >= self.learner_config.replay.param_release_min:
            self._ps_publisher.publish(iteration, message=message)
            self.post_publish()  
            self.exp_counter = 0
'''
    PPO TODOs:
        1) Variant: target network policy WORKS!
        2) RNN -- Make sure its in the same policy. making sure that each data point is homogenous?
        3) Learning rate scheduler 

'''



