import shlex
import sys
import math
from collections import OrderedDict
import surreal.utils as U


class CommandGeneratorBasic:
    def __init__(self, *,
                 num_agents,
                 num_evals,
                 executable,
                 config_commands=None,
                 experiment_folder=None,
                 restore_folder=None):
        """
        Args:
            num_agents: number of agents to run
            num_evals: number of evals to run
            config_py: path to .py executable in the pod
            config_commands (arr): additional commands that pass
                to user-defined config.py, after "--"
            experiment_folder: pass "--experiment-folder <experiment_folder>"
            restore_folder: pass "--restore-folder <restore_folder>"
        """
        self.num_agents = num_agents
        self.num_evals = num_evals
        self.executable = executable

        self.config_commands = config_commands

        self.experiment_folder = experiment_folder
        self.restore_folder = restore_folder

    def get_command(self, role):
        command = ['python', '-u']
        command += [self.config_py]
        command += [role]
        command += ['--']
        if self.config_commands is not None:
            command += self.config_commands
        return ' '.join(command)
