import shlex
import sys
import math
from collections import OrderedDict
import surreal.utils as U


class CommandGenerator:
    def __init__(self, *,
                 num_agents,
                 config_py,
                 experiment_folder,
                 num_evals,
                 batch_agent=1,
                 config_command=None,
                 restore=False,
                 restore_folder=None):
        """
        Args:
            num_agents:
            config_py: remote config.py path in the pod
            experiment_folder: remote session_config.folder in the pod
            config_command: additional commands that pass
                to user-defined config.py, after "--"
            restore: True to restore experiment from checkpoint
            restore_folder: restore experiment from checkpoint folder
        """
        self.num_agents = num_agents
        self.num_evals = num_evals
        self.config_py = config_py
        self.experiment_folder = experiment_folder
        if U.is_sequence(config_command):
            self.config_command = ' '.join(map(shlex.quote, config_command))
        else:
            self.config_command = config_command
        self.restore = restore
        self.restore_folder = restore_folder
        self.batch_agent = batch_agent

    def get_command(self, role):
        command = ['python']
        command += [self.config_py]
        command += [role]
        command += ['--']
        command += ['--experiment-folder', shlex.quote(self.experiment_folder)]
        if self.batch_agent != 1:
            command += ['--agent-batch', str(self.batch_agent)]
        if self.config_command is not None:
            command += [self.config_command]
        if self.restore_folder:
            command += ['--restore-folder', shlex.quote(self.restore_folder)]
        return ' '.join(command)

    def generate(self, save_yaml=None):
        """
        Save __init__ args as well as generated commands to <save_yaml> file

        Args:
            save_yaml: if provided, save the commandline arguments to
                path save_yaml
        """
        cmd_dict = OrderedDict()
        cmd_dict['learner'] = self.get_command('learner')
        if self.batch_agent == 1:
            cmd_dict['agent'] = [self.get_command('agent-{}'.format(i))
                                 for i in range(self.num_agents)]
            cmd_dict['eval'] = [self.get_command('eval-{}'.format(i))
                                for i in range(self.num_evals)]
        else:
            cmd_dict['agent-batch'] = []
            n_batches = math.ceil(float(self.num_agents) / self.batch_agent)
            for i in range(n_batches):
                cmd_dict['agent-batch'].append(
                    self.get_command('agents-{}'.format(i)))

            cmd_dict['eval-batch'] = []
            n_batches_eval = math.ceil(
                float(self.num_evals) / self.batch_agent)
            for i in range(n_batches_eval):
                cmd_dict['eval-batch'].append(
                    self.get_command('evals-{}'.format(i)))

        for role in ['tensorplex',
                     'tensorboard',
                     'loggerplex',
                     'ps',
                     'replay']:
            cmd_dict[role] = self.get_command(role)

        if save_yaml:
            init_dict = OrderedDict()
            for attr in ['num_agents', 'config_py', 'config_command']:
                init_dict[attr] = getattr(self, attr)
            save_dict = OrderedDict(init=init_dict)
            save_dict['commands'] = cmd_dict
            U.yaml_ordered_dump(save_dict, save_yaml)

        print('Launch settings:')
        for attr in ['experiment_folder', 'num_agents',
                     'config_py', 'config_command']:
            print('  {}: {}'.format(attr, getattr(self, attr)))
        return cmd_dict

    # TODO: shall we deprecate
    @staticmethod
    def get_yaml(saved_yaml):
        try:
            return U.EzDict.load_yaml(saved_yaml)['init']
        except FileNotFoundError as e:
            print('Warning:', e, file=sys.stderr)
            return None
