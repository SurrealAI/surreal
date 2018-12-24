"""
Defines the LaunchSettings class that holds all the
information one needs to launch a component of surreal
"""
import time
import os
import sys
import subprocess
from argparse import ArgumentParser
import numpy as np
from tensorplex import Loggerplex
from tensorplex import Tensorplex
from surreal.distributed import ShardedParameterServer
from surreal.replay import ShardedReplay, ReplayLoadBalancer
import surreal.utils as U
import faulthandler
faulthandler.enable()


class Launcher:
    """
        Launchers are shared entrypoint for surreal experiments.
        Launchers define a main function that takes commandline
        arguments in the following way.
        `python launch_ppo.py <component_name> -- [additional_args]`
        component_name defines which part of the experiment should be
        run in this process
        [additional_args] should be shared among all involved processes
        to define behavior globally
    """
    def main(self):
        """
        The main function to be called
        ```
        if __name__ == '__main__':
            launcher = Launcher()
            launcher.main()
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
        parser.add_argument('component_name',
                            type=str,
                            help='which component to launch')
        args = parser.parse_args(parser_args)

        self.config_args = config_args

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
    """
        The default launcher instance of surreal.
    """
    def __init__(self,
                 agent_class,
                 learner_class,
                 replay_class,
                 session_config,
                 env_config,
                 learner_config,
                 eval_mode='eval_stochastic',
                 agent_batch_size=8,
                 eval_batch_size=8,
                 render=False):
        """
        Setup an surreal experiment

        Args:
            agent_class: The Agent subclass to run for agents
            learner_class: The Agent subclass to run for evals
            replay_class: The Replay subclass to run for replays
            session_config: configs passed to all components
            env_config: configs passed to all components
            learner_config: configs passed to all components
            eval_mode: whether evals should be deterministic or
                stochastic. 'eval_deterministic'/'eval_stochastic'
                (default: {'eval_stochastic'})
            agent_batch_size: When running batch_agent,
                how many agents to fork. (default: {8})
            eval_batch_size: When running batch_agent,
                how many evals to fork. (default: {8})
            render: Whether evals should render (default: {False})
        """
        self.agent_class = agent_class
        self.learner_class = learner_class
        self.replay_class = replay_class
        self.session_config = session_config
        self.env_config = env_config
        self.learner_config = learner_config

        self.eval_mode = eval_mode
        self.render = render
        self.agent_batch_size = agent_batch_size
        self.eval_batch_size = eval_batch_size

    def launch(self, component_name_in):
        """
            Launches a surreal experiment

        Args:
            component_name: Allowed components:
                                agent-{*},
                                agents-{*},
                                eval-{*},
                                evals-{*},
                                replay,
                                learner,
                                ps,
                                tensorboard,
        """
        if '-' in component_name_in:
            component_name, component_id = component_name_in.split('-')
            component_id = int(component_id)
        else:
            component_name = component_name_in
            component_id = None

        if component_name == 'agent':
            self.run_agent(agent_id=component_id)
        elif component_name == 'agents':
            agent_ids = self.get_agent_batch(component_id)
            self.run_agent_batch(agent_ids)
        elif component_name == 'eval':
            self.run_eval(eval_id=component_id,
                          mode=self.eval_mode,
                          render=self.render)
        elif component_name == 'evals':
            eval_ids = self.get_eval_batch(component_id)
            self.run_eval_batch(eval_ids, self.eval_mode, self.render)
        elif component_name == 'learner':
            self.run_learner()
        elif component_name == 'ps':
            self.run_ps()
        elif component_name == 'replay':
            self.run_replay()
        elif component_name == 'replay_loadbalancer':
            self.run_replay_loadbalancer()
        elif component_name == 'replay_worker':
            self.run_replay_worker(replay_id=component_id)
        elif component_name == 'tensorboard':
            self.run_tensorboard()
        elif component_name == 'tensorplex':
            self.run_tensorplex()
        elif component_name == 'loggerplex':
            self.run_loggerplex()
        else:
            raise ValueError('Unexpected component {}'.format(component_name))

    def run_component(self, component_name):
        return subprocess.Popen([sys.executable, '-u',
                                 sys.argv[0],
                                 component_name,
                                 '--'] + self.config_args)

    def run_agent(self, agent_id):
        """
            Launches an agent process with agent_id

        Args:
            agent_id (int): agent's id
            iterations (int): if not none, the number of episodes to run before exiting
        """
        agent = self.setup_agent(agent_id)
        agent.main_agent()

    def setup_agent(self, agent_id):
        """'

        Same as launch_agent, but instead of running agent.main()
        infinite loop, returns the agent instance

        Args:
            agent_id: [description]
        """
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

        return agent

    def run_agent_batch(self, agent_ids):
        """
            Launches multiple agent processes with agent_id in agent_ids
            Useful when you want agents to share a GPU

        Args:
            agent_ids (list(int)): each agent's id
        """
        agents = []
        for agent_id in agent_ids:
            component_name = 'agent-{}'.format(agent_id)
            agent = self.run_component(component_name)
            agents.append(agent)
        U.wait_for_popen(agents)

    def get_agent_batch(self, batch_id):
        """
            Returns the agent_ids corresponding to batch_id

        Args:
            batch_id: index of batch

        Returns:
            agent_ids (list): ids of the agents in the batch
        """
        return range(self.agent_batch_size * int(batch_id),
                     self.agent_batch_size * (int(batch_id) + 1))

    def run_eval(self, eval_id, mode, render):
        """
            Launches an eval processes with id eval_id

        Args:
            eval_id (int): eval agent's id
            mode: eval_deterministic or eval_stochastic
            render: see run_eval
        """
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
        """
            Launches multiple eval processes with agent_id in agent_ids
            Useful when you want agents to share a GPU

        Args:
            eval_ids (list(int)): each eval agent's id
            mode:
            render: see run_eval
        """
        evals = []
        for eval_id in eval_ids:
            component_name = 'eval-{}'.format(eval_id)
            agent = self.run_component(component_name)
            evals.append(agent)
        U.wait_for_popen(evals)

    def get_eval_batch(self, batch_id):
        """
            Returns the eval_ids corresponding to batch_id

        Args:
            batch_id: index of batch

        Returns:
            eval_ids (list): ids of the agents in the batch
        """
        return range(self.eval_batch_size * int(batch_id),
                     self.eval_batch_size * int(batch_id) + 1)

    def run_learner(self, iterations=None):
        """
            Launches the learner process.
            Learner consumes experience from replay
            and publishes experience to parameter server
        """
        learner = self.setup_learner()
        learner.main()

    def setup_learner(self):
        """
            Same as run_learner, but returns the Learner instance
            instead of calling learner.main_loop()
        """
        session_config, learner_config, env_config = \
            self.session_config, self.learner_config, self.env_config

        learner_class = self.learner_class
        learner = learner_class(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config
        )

        return learner

    def run_ps(self):
        """
            Lauches the parameter server process.
            Serves parameters to agents
        """
        ps_config = self.session_config.ps

        server = ShardedParameterServer(shards=ps_config.shards)

        server.launch()
        server.join()

    def run_replay(self):
        """
            Launches the replay process.
            Replay collects experience from agents
            and serve them to learner
        """
        loadbalancer = self.run_component('replay_loadbalancer')
        components = [loadbalancer]
        for replay_id in range(self.learner_config.replay.replay_shards):
            component_name = 'replay_worker-{}'.format(replay_id)
            replay = self.run_component(component_name)
            components.append(replay)
        U.wait_for_popen(components)

    def run_replay_loadbalancer(self):
        """
            Launches the learner and agent facing load balancing proxys
            for replays
        """
        loadbalancer = ReplayLoadBalancer()
        loadbalancer.launch()
        loadbalancer.join()

    def run_replay_worker(self, replay_id):
        """
            Launches a single replay server

        Args:
            replay_id: The id of the replay server
        """
        replay = self.replay_class(self.learner_config,
                                   self.env_config,
                                   self.session_config,
                                   index=replay_id)
        replay.start_threads()
        replay.join()

    def run_tensorboard(self):
        """
            Launches a tensorboard process
        """
        folder = os.path.join(self.session_config.folder, 'tensorboard')
        tensorplex_config = self.session_config.tensorplex
        cmd = ['tensorboard',
               '--logdir', folder,
               '--port', str(tensorplex_config.tensorboard_port)]
        subprocess.call(cmd)

    def run_tensorplex(self):
        """
            Launches a tensorplex process.
            It receives data from multiple sources and
            send them to tensorboard.
        """
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
        """
            Launches a loggerplex server.
            It helps distributed logging.
        """
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
