"""
Defines the LaunchSettings class that holds all the
information one needs to launch a component of surreal
"""
import time
import os
import sys
import subprocess
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Process
from tensorplex import Loggerplex
from tensorplex import Tensorplex
from surreal.distributed.ps import ShardedParameterServer
from surreal.replay import ShardedReplay


class Launcher:
    """
    Shared entrypoint for surreal experiments
    """
    def main(self):
        """
        The main function to be called
        ```
        if __name__ == '__main__':
            Launcher().main()
        ```
        """
        argv = sys.argv[1:]
        parser_args = argv
        config_args = []
        if '--' in argv:
            index = argv.index('--')
            parser_args = argv[:index]
            config_args = argv[index + 1:]
        parser = ArgumentParser(description='launch a surreal component')
        parser.add_argument('component_name', type=str, help='which component to launch')
        args = parser.parse_args(parser_args)

        self.setup(config_args)
        self.launch(args.component_name)

    def launch(self, component_name):
        """Launches the specific component

        Args:
            component_name(str): the process to launch
        """
        raise NotImplementedError

    def setup(self, args):
        """
        Sets up the related states for running components

        Args:
            args: A list of commandline arguments provided.
        """
        pass

class SurrealDefaultLauncher(Launcher):
    def __init__(self,
                 agent_class,
                 learner_class,
                 replay_class,
                 session_config,
                 env_config,
                 learner_config,
                 eval_mode='eval_stochastic',
                 render=False):
        self.agent_class = agent_class
        self.learner_class = learner_class
        self.replay_class = replay_class
        self.session_config = session_config
        self.env_config = env_config
        self.learner_config = learner_config

        self.eval_mode = eval_mode
        self.render = render

    def launch(self, component_name_in):
        """
            Launches a surreal experiment

        Args:
            component_name: Allowed components, agent-{*}, replay[-{*}],
                            eval-{*}, learner, ps, tensorboard

        """
        if '-' in component_name_in:
            component_name, component_id = component_name_in.split('-')
        else:
            component_name = component_name_in
            component_id = None

        if component_name == 'agent':
            self.run_agent(agent_id=component_id)
        elif component_name == 'eval':
            self.run_eval(eval_id=component_id,
                          mode=self.eval_mode,
                          render=self.render)
        elif component_name == 'learner':
            self.run_learner()
        elif component_name == 'ps':
            self.run_ps()
        elif component_name == 'replay':
            self.run_replay()
        elif component_name == 'tensorboard':
            self.run_tensorboard()
        elif component_name == 'tensorplex':
            self.run_tensorplex()
        elif component_name == 'loggerplex':
            self.run_loggerplex()
        else:
            raise ValueError('Unexpected component {}'.format(component_name))
        # TODO: batch agent and eval

    def run_agent(self, agent_id):
        np.random.seed(int(time.time() * 100000 % 100000))

        session_config, learner_config, env_config = \
            self.session_config, self.learner_config, self.env_config
        agent_mode = 'training'

        # env, env_config = make_env(env_config)

        agent = self.agent_class(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            agent_id=agent_id,
            agent_mode=agent_mode,
        )

        agent.main_agent()

    def run_agent_batch(self, agent_ids):
        agents = []
        for agent_id in agent_ids:
            agent = Process(target=self.run_agent, args=[agent_id])
            agent.start()
            agents.append(agent)
        for i, agent in enumerate(agents):
            agent.join()
            raise RuntimeError('Agent {} exited with code {}'
                               .format(i, agent.exitcode))

    def run_eval(self, eval_id, mode, render):
        # mode
        # render

        np.random.seed(int(time.time() * 100000 % 100000))

        session_config, learner_config, env_config = \
            self.session_config, self.learner_config, self.env_config

        # env, env_config = make_env(env_config, 'eval')

        agent_mode = mode
        assert agent_mode != 'training'

        agent_class = self.agent_class
        agent = agent_class(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            agent_id=eval_id,
            agent_mode=agent_mode,
            render=render
        )

        agent.main_eval()

    def run_eval_batch(self, eval_ids, mode, render):
        evals = []
        for eval_id in eval_ids:
            agent = Process(target=self.run_eval, args=[eval_id, mode, render])
            agent.start()
            evals.append(agent)
        for i, agent in enumerate(evals):
            agent.join()
            raise RuntimeError('Eval {} exited with code {}'
                               .format(i, agent.exitcode))

    def run_learner(self):
        session_config, learner_config, env_config = \
            self.session_config, self.learner_config, self.env_config

        learner_class = self.learner_class
        learner = learner_class(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config
        )
        learner.main_loop()

    def run_ps(self):
        ps_config = self.session_config.ps

        server = ShardedParameterServer(config=ps_config)

        server.launch()
        server.join()

    def run_replay(self):
        session_config, learner_config, env_config = \
            self.session_config, self.learner_config, self.env_config
        # env, env_config = make_env(env_config)
        # del env

        sharded = ShardedReplay(learner_config=learner_config,
                                env_config=env_config,
                                session_config=session_config,
                                replay_class=self.replay_class)

        sharded.launch()
        sharded.join()

    def run_tensorboard(self):
        folder = os.path.join(self.session_config.folder, 'tensorboard')
        tensorplex_config = self.session_config.tensorplex
        cmd = ['tensorboard',
               '--logdir', folder,
               '--port', str(tensorplex_config.tensorboard_port)]
        subprocess.call(cmd)

    def run_tensorplex(self):
        folder = os.path.join(self.session_config.folder, 'tensorboard')
        tensorplex_config = self.session_config.tensorplex

        tensorplex = Tensorplex(
            folder,
            max_processes=tensorplex_config.max_processes,
        )

        """
            Tensorboard categories:
                learner/replay/eval: algorithmic level, e.g. reward, ... 
                ***-core: low level metrics, i/o speed, computation time, etc.
                ***-system: Metrics derived from raw metric data in core,
                    i.e. exp_in/exp_out
        """
        (tensorplex
            .register_normal_group('learner')
            .register_indexed_group('agent', tensorplex_config.agent_bin_size)
            .register_indexed_group('eval', 4)
            .register_indexed_group('replay', 10))

        port = os.environ['SYMPH_TENSORPLEX_PORT']
        tensorplex.start_server(port=port)

    def run_loggerplex(self):
        folder = self.session_config.folder
        loggerplex_config = self.session_config.loggerplex

        loggerplex = Loggerplex(
            os.path.join(folder, 'logs'),
            level=loggerplex_config.level,
            overwrite=loggerplex_config.overwrite,
            show_level=loggerplex_config.show_level,
            time_format=loggerplex_config.time_format
        )
        port = os.environ['SYMPH_LOGGERPLEX_PORT']
        loggerplex.start_server(port)

