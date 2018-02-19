import shlex
from collections import OrderedDict
import surreal.utils as U


class CommandGenerator:
    def __init__(self, *,
                 num_agents,
                 config_py,
                 experiment_folder,
                 config_command=None,
                 service_url=None,
                 restore_ckpt=False,
                 restore_ckpt_folder=None):
        """
        Args:
            num_agents:
            config_py: remote config.py path in the pod
            experiment_folder: remote session_config.folder in the pod
            config_command: additional commands that pass to user-defined config.py, after "--"
            service_url: DNS <experiment-name>.surreal
            restore_ckpt: True to restore experiment from checkpoint
            restore_ckpt_folder: restore experiment from checkpoint folder
        """
        self.num_agents = num_agents
        self.config_py = config_py
        self.experiment_folder = experiment_folder
        if U.is_sequence(config_command):
            self.config_command = ' '.join(map(shlex.quote, config_command))
        else:
            self.config_command = config_command
        self.service_url = service_url
        self.restore_ckpt = restore_ckpt
        self.restore_ckpt_folder = restore_ckpt_folder

    def get_command(self, role, args=None):
        if args is None:
            args = []
        command = ['python -u -m', 'surreal.main_scripts.runner', self.config_py]
        command += ['--experiment-folder', shlex.quote(self.experiment_folder)]
        if self.service_url is not None:
            command += ['--service-url', shlex.quote(self.service_url)]
        command += [role]
        command += args
        if self.config_command is not None:
            command += ['--', self.config_command]
        return ' '.join(command)

    def generate(self, save_yaml=None):
        """
        Save __init__ args as well as generated commands to <save_yaml> file
        """
        cmd_dict = OrderedDict()
        if self.restore_ckpt:
            restore_cmd = ['--restore']
            if self.restore_ckpt_folder:
                restore_cmd += ['--restore-folder',
                                shlex.quote(self.restore_ckpt_folder)]
        else:
            restore_cmd = []
        cmd_dict['learner'] = self.get_command('learner', restore_cmd)

        cmd_dict['agent'] = [self.get_command('agent', [str(i)])
                             for i in range(self.num_agents)]
        cmd_dict['eval'] = [self.get_command('eval', ['--mode', 'deterministic', '--id', '0'])]

        for role in ['tensorplex', 'tensorboard', 'loggerplex', 'ps', 'replay']:
            cmd_dict[role] = self.get_command(role)

        if save_yaml:
            U.f_mkdir_in_path(save_yaml)
            init_dict = OrderedDict()
            for attr in ['config_py', 'config_command', 'service_url']:
                init_dict[attr] = getattr(self, attr)
            save_dict = OrderedDict(init=init_dict)
            save_dict['commands'] = cmd_dict
            U.yaml_ordered_dump(save_dict, save_yaml)

        return cmd_dict

    @classmethod
    def reconstruct(cls, saved_yaml, *,  # kwargs after this are for init
                    experiment_folder, num_agents,
                    restore_ckpt_folder, restore_ckpt=True,
                    **kwargs):
        """
        CommandGenerator init args from saved yaml
        kwargs can override any saved init_args
        """
        init_kwargs = U.yaml_load(saved_yaml)['init']
        init_kwargs.update(dict(
            num_agents=num_agents,
            experiment_folder=experiment_folder,
            restore_ckpt=restore_ckpt,
            restore_ckpt_folder=restore_ckpt_folder
        ))
        init_kwargs.update(kwargs)
        return cls(**init_kwargs)


if __name__ == "__main__":
    gen = CommandGenerator(config_py='~/.abc',
                           num_agents=3,
                           experiment_folder='/remote/exp/folder',
                           config_command='--test-command def -m "ab cd ef"',
                           service_url='foo.bar.com',
                           restore_ckpt=True,
                           restore_ckpt_folder='remote/restore/folder')
    save_yaml = '~/Temp/kurreal/launch_commands.yml'
    di = gen.generate(save_yaml)
    print(di)
    print(di['learner'])

    gen = CommandGenerator.reconstruct(save_yaml,
                                       experiment_folder='/new/exp/foder',
                                       num_agents=4,
                                       restore_ckpt=True,
                                       restore_ckpt_folder='my/restore/',
                                       service_url='this.changed.yo')
    print(vars(gen))
