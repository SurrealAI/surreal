import os
from copy import copy
from symphony.commandline import SymphonyParser
from symphony.engine import SymphonyConfig, Cluster
from benedict import BeneDict
from surreal.launch import (
    CommandGenerator,
    setup_network,
)
import surreal.utils as U


def _merge_setting_dictionaries(customize, base):
    di = copy(base)
    for key in di:
        if isinstance(di[key], dict):
            if key in customize:
                di[key] = _merge_setting_dictionaries(customize[key], di[key])
        else:
            if key in customize and customize[key] is not None:
                di[key] = customize[key]
    return di


class TurrealParser(SymphonyParser):
    def create_cluster(self):
        return Cluster.new('tmux')

    def setup(self):
        super().setup()
        self.config = BeneDict()
        self.load_config()
        self._setup_create()

    def load_config(self):
        surreal_yml_path = U.get_config_file()
        if not U.f_exists(surreal_yml_path):
            raise ValueError('Cannot find surreal config file at {}'
                             .format(surreal_yml_path))
        self.config = BeneDict.load_yaml_file(surreal_yml_path)
        SymphonyConfig().set_username(self.username)
        SymphonyConfig().set_experiment_folder(self.folder)

    @property
    def folder(self):
        if 'tmux_results_folder' not in self.config:
            raise KeyError('Please specify "tmux_results_folder" in ~/.surreal.yml')
        return U.f_expand(self.config.tmux_results_folder)

    @property
    def username(self):
        if 'username' not in self.config:
            raise KeyError('Please specify "username" in ~/.surreal.yml')
        return self.config.username

    def _setup_create(self):
        parser = self.add_subparser('create', aliases=['c'])
        self._add_experiment_name(parser)
        parser.add_argument(
            '--algorithm',
            type=str,
            default='ppo',
            help='ddpg / ppo or the location of algorithm python script'
        )
        parser.add_argument(
            '--num_agents',
            type=int,
            default=2,
            help='number of agent pods to run in parallel.'
        )
        parser.add_argument(
            '--num_evals',
            type=int,
            default=1,
            help='number of eval pods to run in parallel.'
        )
        parser.add_argument(
            '--env',
            type=str,
            default='gym:HalfCheetah-v2',
            help='What environment to run'
        )
        self._add_dry_run(parser)

    # ==================== helpers ====================
    def _add_dry_run(self, parser):
        parser.add_argument(
            '-dr', '--dry-run',
            action='store_true',
            help='print the kubectl command without actually executing it.'
        )

    def _process_experiment_name(self, experiment_name):
        """
        experiment_name will be used as DNS, so must not have underscore or dot
        """
        new_name = experiment_name.lower().replace('.', '-').replace('_', '-')
        if new_name != experiment_name:
            print('experiment name string has been fixed: {} -> {}'
                  .format(experiment_name, new_name))
        return new_name

    def action_create(self, args):
        """
            Spin up a multi-node distributed Surreal experiment.
            Put any command line args that pass to the config script after "--"
        """
        cluster = self.create_cluster()
        experiment_name = self._process_experiment_name(args.experiment_name)
        exp = cluster.new_experiment(
            experiment_name,
            preamble_cmds=self.config.tmux_preamble_cmds)

        algorithm_args = args.remainder
        algorithm_args += [
            "--num-agents",
            str(args.num_agents),
            ]
        experiment_folder = os.path.join(self.folder, experiment_name)
        print('Writing experiment output to {}'.format(experiment_folder))
        algorithm_args += ["--experiment-folder",
                           experiment_folder]
        algorithm_args += ["--env", args.env]
        executable = self._find_executable(args.algorithm)
        cmd_gen = CommandGenerator(
            num_agents=args.num_agents,
            num_evals=args.num_evals,
            executable=executable,
            config_commands=algorithm_args)

        learner = exp.new_process(
            'learner',
            cmds=[cmd_gen.get_command('learner')])
        # learner.set_env('DISABLE_MUJOCO_RENDERING', "1")

        replay = exp.new_process(
            'replay',
            cmds=[cmd_gen.get_command('replay')])

        ps = exp.new_process(
            'ps',
            cmds=[cmd_gen.get_command('ps')])

        tensorboard = exp.new_process(
            'tensorboard',
            cmds=[cmd_gen.get_command('tensorboard')])

        tensorplex = exp.new_process(
            'tensorplex',
            cmds=[cmd_gen.get_command('tensorplex')])

        loggerplex = exp.new_process(
            'loggerplex',
            cmds=[cmd_gen.get_command('loggerplex')])

        agents = []
        for i in range(args.num_agents):
            agent_name = 'agent-{}'.format(i)
            agent = exp.new_process(
                agent_name,
                cmds=[cmd_gen.get_command(agent_name)])
            agents.append(agent)

        evals = []
        for i in range(args.num_evals):
            eval_name = 'eval-{}'.format(i)
            eval_p = exp.new_process(
                eval_name,
                cmds=[cmd_gen.get_command(eval_name)])
            evals.append(eval_p)

        setup_network(agents=agents,
                      evals=evals,
                      learner=learner,
                      replay=replay,
                      ps=ps,
                      tensorboard=tensorboard,
                      tensorplex=tensorplex,
                      loggerplex=loggerplex)

        cluster.launch(exp, dry_run=args.dry_run)

    def _find_executable(self, name):
        """
            Finds the .py file corresponding to the algorithm specified

        Args:
            name: ddpg / ppo / <path in container to compatible .py file>
        """
        if name == 'ddpg':
            return 'surreal-ddpg'
        elif name == 'ppo':
            return 'surreal-ppo'
        else:
            return name


def main():
    TurrealParser().main()


if __name__ == '__main__':
    main()
