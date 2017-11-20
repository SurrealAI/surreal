from .tmux_runner import TmuxRunner
from .config import Config
import time
import collections


class TmuxCluster(object):
    """
    Launch the following in order:
    1. Redis replay server
    2. Redis parameter server
    3. Redis tensorplex/loggerplex server
    4. Learner
    5. Evaluator (=None to skip evaluation)
    6. Tensorboard
    7. Army of agents
    """
    def __init__(self, *,
                 cluster_name,
                 session_config,
                 agent_script,
                 learner_script,
                 evaluator_script,
                 start_dir='.',
                 dry_run=False
                 ):
        """
        Args:
            session_config:
            agent_args: list of list of command line args OR command strings.
                Each agent might have a different command line invocation,
                such as different names and exploration strategies. E.g.
                [
                    ['--explore', 'strategy1', '--id', 10],
                    ['--explore', 'strategy2', '--id', 13, '--anneal', 0.5],
                    '--explore strat3 --id 22'  # or simply a long string
                ]
        """
        self.config = Config(session_config)
        self.agent_cmd = self._get_python_cmd(agent_script)
        self.learner_cmd = self._get_python_cmd(learner_script)
        if evaluator_script is None:
            self.evaluator_cmd = None
        else:
            self.evaluator_cmd = self._get_python_cmd(evaluator_script)
        self.redis_session = 'redis-' + cluster_name
        self.agent_session = 'agent-' + cluster_name
        self.learner_session = 'learner-' + cluster_name
        self._tmux = TmuxRunner(
            start_dir=start_dir,
            verbose=True,
            dry_run=dry_run
        )

    def _get_python_cmd(self, python_script):
        if ' python ' in python_script:
            return python_script  # already a command
        if not python_script.endswith('.py') and '/' in python_script:
            raise ValueError('Ill-formed python script ' + python_script +
                             ' should be either pkg1.pkg2.myscript or '
                             'pkg1/pkg2/myscript.py')
        if python_script.endswith('.py'):
            return 'python -u ' + python_script
        else:
            # python -m surreal.main.run_cartpole
            return 'python -u -m ' + python_script

    def _get_agent_info(self, agent_names, agent_args_):
        assert isinstance(agent_names, list)
        assert isinstance(agent_args_, list)
        agent_names = [str(_name) for _name in agent_names]
        assert len(agent_names) == len(set(agent_names)), \
            'must not duplicate agent names'
        assert len(agent_names) == len(agent_args_)
        agent_args = []
        for cmd_args in agent_args_:
            if cmd_args is None:
                cmd = ''
            elif isinstance(cmd_args, str):
                cmd = cmd_args
            elif isinstance(cmd_args, list):
                cmd = ' '.join(str(x) for x in cmd_args)
            else:
                raise ValueError('Must be a list of command line arg list '
                                 'OR a list of command strings.')
            agent_args.append(cmd)
        return agent_names, agent_args

    def _session_group(self, group):
        assert group in ['agent', 'learner', 'redis']
        return {
            'agent': self.agent_session,
            'learner': self.learner_session,
            'redis': self.redis_session
        }[group]

    def is_launched(self, group):
        return self._session_group(group) in self._tmux.list_session_names()

    def running_agents(self):
        return self._tmux.list_window_names(self.agent_session)

    def launch(self, agent_names, agent_args):
        # Redis session
        if not self.is_launched('redis'):
            for window_name, port in [
                (self.config.replay.name, self.config.replay.port),
                (self.config.ps.name, self.config.ps.port),
                ('tensorplex', self.config.tensorplex.port),
            ]:
                self._tmux.run(
                    session_name=self.redis_session,
                    window_name=window_name,
                    cmd='redis-server --port {} --protected-mode no'.format(port)
                )
        # Learner session
        if not self.is_launched('learner'):
            self._tmux.run(
                session_name=self.learner_session,
                window_name='learner',
                cmd=self.learner_cmd
            )
            if self.evaluator_cmd is not None:
                self._tmux.run(
                    session_name=self.learner_session,
                    window_name='evaluator',
                    cmd=self.evaluator_cmd
                )
            # TODO launch tensorboard
        # Agent session
        if not self.is_launched('agent'):
            self.add_agents(agent_names, agent_args)

    def add_agents(self, agent_names, agent_args):
        agent_names, agent_args = self._get_agent_info(
            agent_names, agent_args
        )
        # should not duplicate agent name
        assert not (set(self.running_agents()) & set(agent_names)), \
            'some agents already running, cannot launch duplicates.'
        for agent_name, args in zip(agent_names, agent_args):
            self._tmux.run(
                session_name=self.agent_session,
                window_name=agent_name,
                cmd=self.agent_cmd + ' ' + args
            )

    def kill_agents(self, agent_names):
        assert self.is_launched('agent'), 'agents not yet launched'
        for name in agent_names:
            self._tmux.kill(
                session_name=self.agent_session,
                window_name=str(name)
            )

    def _iterate_all_windows(self):
        for sess in [self.agent_session,
                     self.learner_session,
                     self.redis_session]:
            for win in self._tmux.list_window_names(sess):
                yield sess, win

    def get_stdout(self, group=None, window=None, history=0):
        """
        Args:
            group: [agent, learner, tensorplex] None for all
            window: get specific window. None for all windows.
                If group is None, window must also be None.
        Returns:
            stdout string if both `group` and `window` are specified.
            else OrderedDict({"session:window": "pane stdout"})
            pane stdout captures only the visible part unless you specify history
        """
        if group is None:
            assert window is None
        else:
            group = self._session_group(group)
        if group and window:
            stdout = self._tmux.get_stdout(group, str(window), history=history)
            return '\n'.join(stdout)

        outdict = collections.OrderedDict()
        for sess, win in self._iterate_all_windows():
            if group and group != sess:
                continue
            stdout = self._tmux.get_stdout(sess, win, history=history)
            assert isinstance(stdout, list)  # list of lines
            outdict['{}:{}'.format(sess, win)] = '\n'.join(stdout)
        return outdict

    def check_error(self):
        """
        Returns:
            OrderedDict({"session:window": "error-message"})
        """
        outdict = collections.OrderedDict()
        for sess, win in self._iterate_all_windows():
            err = self._tmux.check_error(sess, win)
            if err:
                outdict['{}:{}'.format(sess, win)] = err
        return outdict

    @property
    def num_agents(self):
        return len(self.agent_names)

    def killall(self):
        self._tmux.kill(self.agent_session)
        self._tmux.kill(self.learner_session)
        self._tmux.kill(self.redis_session)

