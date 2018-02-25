import time
import json
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from surreal.session import (PeriodicTracker, get_tensorplex_client)
import surreal.utils as U
from .wrapper import Wrapper


class EpisodeMonitor(Wrapper):
    """
    Access the public properties to get episode history:
    - episode_rewards
    - episode_steps
    - total_steps
    - episode_durations (in seconds)
    - num_episodes == len(episode_rewards)
    """
    def __init__(self, env):
        super().__init__(env)
        self._tstart_ep0 = time.time()
        self._tstart_current_ep = None
        self._rewards_current_ep = None
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_durations = []
        self.total_steps = 0

    def _reset(self, **kwargs):
        self._rewards_current_ep = []
        self._tstart_current_ep = time.time()
        return self.env.reset(**kwargs)

    def _step(self, action):
        ob, rew, done, info = self.env.step(action)
        self._rewards_current_ep.append(rew)
        if done:
            eprew = round(sum(self._rewards_current_ep), 6)
            epsteps = len(self._rewards_current_ep)
            eptime = round(time.time() - self._tstart_current_ep, 6)
            epinfo = {
                "reward": eprew,
                "steps": epsteps,
                "duration": eptime,
                "total_elapsed": round(time.time() - self._tstart_ep0, 6),
            }
            self.episode_rewards.append(eprew)
            self.episode_steps.append(epsteps)
            self.episode_durations.append(eptime)
            info['episode'] = epinfo
        self.total_steps += 1
        return ob, rew, done, info

    @property
    def num_episodes(self):
        return len(self.episode_rewards)

    def step_per_sec(self, average_episodes):
        """
        Speedometer, step per second

        Args:
            average_episodes: average over the past N episodes
        """
        assert average_episodes > 0
        return (sum(self.episode_steps[-average_episodes:])
                / (sum(self.episode_durations[-average_episodes:]) + 1e-7))


class ConsoleMonitor(EpisodeMonitor):
    def __init__(self, env,
                 update_interval=10,
                 average_over=10,
                 extra_rows=None):
        """
        Args:
            update_interval: print every N episodes
            average_over: average rewards/speed over the last N episodes
            extra_rows: an OrderedDict {'row caption': function(total_steps, num_episodes)}
                to generate extra rows to the printed table.
        """
        super().__init__(env)
        self._periodic = PeriodicTracker(update_interval)
        self._avg = average_over
        if extra_rows is None:
            self._extra_rows = OrderedDict()
        else:
            assert isinstance(extra_rows, OrderedDict), \
                'extra_rows spec {"row caption": function(total_steps, ' \
                'num_episodes)} must be an OrderedDict'
            self._extra_rows = extra_rows

    def _step(self, action):
        ob, r, done, info = super()._step(action)
        if done and self._periodic.track_increment():
            info_table = []
            avg_reward = U.mean(self.episode_rewards[-self._avg:])
            info_table.append(['Last {} rewards'.format(self._avg),
                               U.fformat(avg_reward, 3)])
            avg_speed = self.step_per_sec(self._avg)
            info_table.append(['Speed iter/s',
                               U.fformat(avg_speed, 1)])
            info_table.append(['Total steps', self.total_steps])
            info_table.append(['Episodes', self.num_episodes])
            for row_caption, row_func in self._extra_rows.items():
                row_value = row_func(self.total_steps, self.num_episodes)
                info_table.append([row_caption, str(row_value)])
            # `fancy_grid` doesn't work in terminal that doesn't display unicode
            print(tabulate(info_table, tablefmt='simple', numalign='left'))
        return ob, r, done, info


class TrainingTensorplexMonitor(EpisodeMonitor):
    def __init__(self, env,
                 agent_id,
                 session_config,
                 separate_plots=True):
        """
        Display "reward" and "step_per_s" curves on Tensorboard

        Args:
            env:
            agent_id: int.
            session_config: to construct AgentTensorplex
            - interval: log to Tensorplex every N episodes.
            - average_episodes: average rewards/speed over the last N episodes
            separate_plots: True to put reward plot in a separate section on
                Tensorboard, False to put all plots together
        """
        super().__init__(env)
        U.assert_type(agent_id, int)
        self.tensorplex = get_tensorplex_client(
            '{}/{}'.format('agent', agent_id),
            session_config
        )
        interval = session_config['tensorplex']['update_schedule']['training_env']
        self._periodic = PeriodicTracker(interval)
        self._avg = interval
        self._separate_plots = separate_plots

    def _get_tag(self, tag):
        if self._separate_plots:
            return ':' + tag  # see Tensorplex tag semantics
        else:
            return tag

    def _step(self, action):
        ob, r, done, info = super()._step(action)
        if done and self._periodic.track_increment():
            scalar_values = {
                self._get_tag('reward'):
                    U.mean(self.episode_rewards[-self._avg:]),
                'step_per_s':
                    self.step_per_sec(self._avg),
            }
            self.tensorplex.add_scalars(
                scalar_values,
                global_step=self.num_episodes
            )
        return ob, r, done, info


class EvalTensorplexMonitor(EpisodeMonitor):
    def __init__(self, env,
                 eval_id,
                 fetch_parameter,
                 session_config,
                 separate_plots=False):
        """
        Display "reward" and "step_per_s" curves on Tensorboard

        Args:
            env:
            eval_id:
            fetch_parameter: lambda function that pulls from parameter server
            session_config: to construct AgentTensorplex
            - interval: log to Tensorplex every N episodes.
            - average_episodes: average rewards/speed over the last N episodes
            separate_plots: True to put reward plot in a separate section on
                Tensorboard, False to put all plots together
        """
        super().__init__(env)
        self.tensorplex = get_tensorplex_client(
            '{}/{}'.format('eval', eval_id),
            session_config
        )
        interval = session_config['tensorplex']['update_schedule']['eval_env']
        self._periodic = PeriodicTracker(interval)
        self._avg = interval
        self._separate_plots = separate_plots
        self._throttle_sleep = \
            session_config['tensorplex']['update_schedule']['eval_env_sleep']
        self._fetch_parameter = fetch_parameter
        self._fetch_parameter()  # if this eval is late to the party

    def _get_tag(self, tag):
        if self._separate_plots:
            return ':' + tag  # see Tensorplex tag semantics
        else:
            return tag

    def _step(self, action):
        ob, r, done, info = super()._step(action)
        if done and self._periodic.track_increment():
            scalar_values = {
                self._get_tag('reward'):
                    U.mean(self.episode_rewards[-self._avg:]),
                'step_per_s':
                    self.step_per_sec(self._avg),
            }
            self.tensorplex.add_scalars(
                scalar_values,
                global_step=self.num_episodes
            )
            time.sleep(self._throttle_sleep)
            self._fetch_parameter()
        return ob, r, done, info
