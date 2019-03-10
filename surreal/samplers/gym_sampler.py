"""
This file contains a wrapper for sampling environment states
from a set of demonstrations on every reset. The main use case is for
altering the start state distribution of training episodes for
learning RL policies.
"""

import random
import os
import numpy as np


class GymSampler(object):
    env = None

    def __init__(
        self,
        demo_path,
        num_traj=-1,
        sampling_schemes=["uniform"],
    ):
        """
        Initializes a wrapper that provides support for resetting the environment
        state to one from a demonstration. It also supports curriculums for
        altering how often to sample from demonstration vs. sampling a reset
        state from the environment.

        Args:
            demo_path (string): The path to the folder containing the demonstrations.
                It should contain `.npy` files, each corresponding to a demonstration

            num_traj (int): If provided, subsample @number demonstrations from the
                provided set of demonstrations instead of using all of them.

            sampling_schemes (list of strings): A list of sampling schemes
                to be used. The following strings are valid schemes:

                    "uniform" : sample a state from a demonstration uniformly at random

        """

        self.demo_path = demo_path

        # list of all demonstration and action file paths
        self.demo_list = self._list_file_type("observation")
        self.action_list = self._list_file_type("action")
        self.num_demonstrations = len(self.demo_list)

        # subsample a selection of demonstrations if requested
        if num_traj > 0:
            random.seed(3141)  # ensure that the same set is sampled every time
            self.demo_list = random.sample(self.demo_list, num_traj)

        self.demo_sampled = 0

        self.sample_method_dict = {
            "uniform": "_uniform_sample",
        }

        self.sampling_schemes = sampling_schemes

        # make sure the list of schemes is valid
        schemes = self.sample_method_dict.keys()
        assert np.all([(s in schemes) for s in self.sampling_schemes])

    def _list_file_type(self, starts_with):
        files = [os.path.join(self.demo_path, f) for f in os.listdir(self.demo_path) if
                 f.endswith(".npy") and f.startswith(starts_with)]
        return sorted(files)

    def sample(self):
        """
        This is the core sampling method. Samples a state from a
        demonstration, in accordance with the configuration.
        Right now the uniform sampling method is always selected.
        """

        sample_method = getattr(self, self.sample_method_dict[self.sampling_schemes[0]])
        return sample_method()

    def _uniform_sample(self):
        """
        Sampling method.

        First uniformly sample a demonstration from the set of demonstrations.
        Then uniformly sample a state from the selected demonstration.
        """
        episode_choice = random.randint(0, self.num_demonstrations - 1)
        # get a random episode index
        demo_episode_path = self.demo_list[episode_choice]
        action_episode_path = self.action_list[episode_choice]

        # select a flattened mujoco demonstration and
        # action step uniformly from this episode
        demonstration_episode = np.load(demo_episode_path)
        step_choice = random.randint(0, demonstration_episode.shape[0] - 1)
        demonstration_state = demonstration_episode[step_choice]
        action_episode = np.load(action_episode_path)
        action_state = action_episode[step_choice]

        return demonstration_state, action_state
