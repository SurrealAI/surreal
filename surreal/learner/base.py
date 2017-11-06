"""
Template class for all learners
"""
import surreal.utils as U
from easydict import EasyDict


class Learner(object):
    def __init__(self, config, model):
        """

        Args:
            config: a dictionary of hyperparameters. It can include a special
                section "log": {logger configs}
            model: utils.pytorch.Module for the policy network
        """
        U.assert_type(config, dict)
        U.assert_type(model, U.Module)
        self.config = EasyDict(config)
        self.model = model
        log_kwargs = self.config['log'] if 'log' in self.config else {}
        self.log = U.Logger.get_logger('Learner', **log_kwargs)

    def learn(self, batch_exp, batch_i):
        """
        Abstract method runs one step of learning

        Args:
            batch_exp: batched experience, can be a tuple of pytorch-ready
                tensor objects (obs_t, obs_t+1, rewards, actions, dones)
            batch_i: number of batches processed so far

        Returns:
            td_error or other values for prioritized replay
        """
        raise NotImplementedError

