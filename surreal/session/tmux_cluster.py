from .tmux_runner import TmuxRunner
from .config import Config
from .default_configs import BASE_SESSION_CONFIG
import time
import json
import shlex
from collections import OrderedDict
import surreal.utils as U


class TmuxCluster(object):
    """
    Launch the following in order:
    1. Redis replay server
    2. Redis parameter server
    3. Redis tensorplex/loggerplex server
    4. Loggerplex (distributed logging server script)
    5. Tensorplex (distributed tensorplex server script)
    6. Tensorboard, `tensorboard --logdir . --port <tensorboard_port>`
    7. Learner
    8. Evaluator (=None to skip evaluation)
    9. Army of agents
    """
    # TODO support multiple eval threads
    LOGGERPLEX_SCRIPT = 'surreal.session.run_loggerplex_server'
    TENSORPLEX_SCRIPT = 'surreal.session.run_tensorplex_server'

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
        self.config = Config(session_config).extend(BASE_SESSION_CONFIG)
        self.agent_cmd = self._get_python_cmd(agent_script)
        self.learner_cmd = self._get_python_cmd(learner_script)
        if evaluator_script is None:
            self.evaluator_cmd = None
        else:
            self.evaluator_cmd = self._get_python_cmd(evaluator_script)
        self.infras_session = 'infras-' + cluster_name
        self.agent_session = 'agent-' + cluster_name
        self.learner_session = 'learner-' + cluster_name
        self._tmux = TmuxRunner(
            start_dir=start_dir,
            verbose=True,
            dry_run=dry_run
        )

    def _get_python_cmd(self, python_script):
        if python_script.startswith('python'):
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
        U.assert_type(agent_names, list)
        U.assert_type(agent_args_, list)
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
        assert group in ['agent', 'learner', 'infras']
        return {
            'agent': self.agent_session,
            'learner': self.learner_session,
            'infras': self.infras_session
        }[group]

    def is_launched(self, group):
        return self._session_group(group) in self._tmux.list_session_names()

    def get_running_agents(self):
        return self._tmux.list_window_names(self.agent_session)

    def _get_tensorplex_cmd(self, script):
        script = self._get_python_cmd(script)
        # dump config to JSON as command line arg
        return script + ' ' + shlex.quote(json.dumps(self.config))

    def launch(self, agent_names, agent_args):
        # Infrastructure session
        if not self.is_launched('infras'):
            for window_name, port in [
                (self.config.replay.name, self.config.replay.port),
                (self.config.ps.name, self.config.ps.port),
                ('plex', self.config.tensorplex.port),
            ]:
                self._tmux.run(
                    session_name=self.infras_session,
                    window_name='redis-'+window_name,
                    cmd='redis-server --port {} --protected-mode no'.format(port)
                )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='loggerplex',
                cmd=self._get_tensorplex_cmd(self.LOGGERPLEX_SCRIPT)
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='tensorplex',
                cmd=self._get_tensorplex_cmd(self.TENSORPLEX_SCRIPT)
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='tensorboard',
                cmd='tensorboard --logdir {} --port {}'.format(
                    self.config.folder,
                    self.config.tensorplex.tensorboard_port
                )
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
                    window_name='eval',
                    cmd=self.evaluator_cmd
                )
        # Agent session
        if not self.is_launched('agent'):
            self.add_agents(agent_names, agent_args)

    def add_agents(self, agent_names, agent_args):
        agent_names, agent_args = self._get_agent_info(
            agent_names, agent_args
        )
        # should not duplicate agent name
        assert not (set(self.get_running_agents()) & set(agent_names)), \
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
                     self.infras_session]:
            for win in self._tmux.list_window_names(sess):
                yield sess, win

    def _iterated_filtered_windows(self, group=None, window=None):
        if group is None:
            assert window is None, \
                'when group is specified, window should be left as None'
            yield from self._iterate_all_windows()
        else:
            sess = self._session_group(group)
            if window is None:
                for win in self._tmux.list_window_names(sess):
                    yield sess, win
            else:
                window = str(window)
                yield sess, window  # just one iteration

    def list_windows(self):
        windict = OrderedDict()
        for sess, win in self._iterate_all_windows():
            if sess in windict:
                windict[sess].append(win)
            else:
                windict[sess] = [win]
        return windict

    def get_stdout(self, group=None, window=None, history=0):
        """
        Args:
            group: [agent, learner, infras] None for all
            window: get specific window. None for all windows.
                If group is None, window must also be None.
        Returns:
            OrderedDict({"session:window": "pane stdout"})
            pane stdout captures only the visible part unless you specify history
        """
        outdict = OrderedDict()
        for sess, win in self._iterated_filtered_windows(group, window):
            stdout = self._tmux.get_stdout(sess, win, history=history)
            U.assert_type(stdout, list)  # internal
            outdict['{}:{}'.format(sess, win)] = '\n'.join(stdout)
        return outdict

    def print_stdout(self, group=None, window=None, history=0, sep='='):
        """
        Args:
            group:
            window:
            history:
            sep: separator symbol "=" in "====== <win name> ======"
        """
        for win, out in self.get_stdout(
                group=group,
                window=window,
                history=history
        ).items():
            print(sep*20, win, sep*20)
            print(out)

    def check_error(self, group=None, window=None):
        """
        For group and window semantics, refer to `get_stdout`

        Returns:
            OrderedDict({"session:window": "error-message"})
        """
        outdict = OrderedDict()
        for sess, win in self._iterated_filtered_windows(group, window):
            err = self._tmux.check_error(sess, win, history=100)
            if err:
                outdict['{}:{}'.format(sess, win)] = err
        return outdict

    def print_error(self, group=None, window=None, sep='='):
        errdict = self.check_error(group, window)
        if len(errdict) == 0:
            if group is None and window is None:
                print('No error found in cluster.')
            elif window is None:
                print('No error found in group "{}" (tmux session "{}").'
                      .format(group, self._session_group(group)))
            else:
                print('No error found in "{}:{}" window'
                      .format(self._session_group(group), window))
        else:
            for win, out in errdict.items():
                print(sep*20, win, sep*20)
                print(out)

    def killall(self):
        self._tmux.kill(self.agent_session)
        self._tmux.kill(self.learner_session)
        self._tmux.kill(self.infras_session)

