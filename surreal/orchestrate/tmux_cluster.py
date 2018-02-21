from .tmux_runner import TmuxRunner
from surreal.session.config import Config
from surreal.session.default_configs import BASE_SESSION_CONFIG
import time
import json
import shlex
from collections import OrderedDict
import surreal.utils as U
from surreal.main_scripts.runner import load_config

class TmuxCluster(object):
    """
    Launch the following in order:
    1. Loggerplex (distributed logging server script)
    2. Tensorplex (distributed tensorplex server script)
    3. Tensorboard, `tensorboard --logdir . --port <tensorboard_port>`
    4. Parameter server (standalone script)
    5. Replay server
    6. Learner
    7. Evaluator (=None to skip evaluation)
    8. Army of agents
    """
    def __init__(self,
                 cluster_name,
                 config_path,
                 experiment_folder,
                 start_dir='.',
                 preamble_cmd=None,
                 config_command=None,
                 dry_run=False,
                 ):
        """
        Args:
            @config_path: File system path to a .py config file TODO: Add more documentation for config files.
            For now look at the example of surreal/main/ddpg_configs.py
            @start_dir: Tmux initial directory
            @preamble_cmd: Commands to run in a tmux window before running the surreal process
            E.g. source activate [name of your virtual env]
            @config_command: Command to supply to the config file through the runner's --config-command argument
            Will be escaped by shlex.quote
            @dry_run: Set for tmux
        """
        self.all_config = load_config(config_path, shlex.split(config_command))
        self.config = self.all_config.session_config
        self.config_path = config_path
        self.config_command = config_command
        # TODO: This is very error prone, we need to fix it
        self.experiment_folder = experiment_folder
        self.config.folder = experiment_folder

        self.infras_session = 'infras-' + cluster_name
        self.agent_session = 'agent-' + cluster_name
        self.learner_session = 'learner-' + cluster_name
        self._tmux = TmuxRunner(
            start_dir=start_dir,
            preamble_cmd=preamble_cmd,
            verbose=True,
            dry_run=dry_run
        )

    def get_command(self, mode, args=None):
        """
            mode is agent/learner/...
            args is the surreal defined argument to give to agent/learner, in a string!!!!
        """
        command = ['python -u -m', 'surreal.main_scripts.runner', self.config_path]
        command += ['--experiment-folder', self.experiment_folder]
        command += [mode]
        if args is not None:
            command += [args]
        if self.config_command is not None:
            command += ['--', self.config_command]
        return ' '.join(command)

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

    def get_running_evals(self):
        return [win for win
                in self._tmux.list_window_names(self.learner_session)
                if not win.startswith('learner')]

    def _get_cmd_with_json(self, script):
        script = self._get_python_cmd(script)
        # dump config to JSON as command line arg
        return script + ' ' + shlex.quote(json.dumps(self.config))

    def launch(self,
               agent_names,
               agent_args,
               eval_names=None,
               eval_args=None):
        # Infrastructure session
        if not self.is_launched('infras'):
            self._tmux.run(
                session_name=self.infras_session,
                window_name='loggerplex',
                cmd=self.get_command('loggerplex')
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='tensorplex',
                cmd=self.get_command('tensorplex')
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='tensorboard',
                cmd=self.get_command('tensorboard')
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='ps',
                cmd=self.get_command('ps')
            )
        # Learner session
        if not self.is_launched('learner'):
            self._tmux.run(
                session_name=self.learner_session,
                window_name='replay',
                cmd=self.get_command('replay')
            )
            self._tmux.run(
                session_name=self.learner_session,
                window_name='learner',
                cmd=self.get_command('learner')
            )
            
        self.add_evals(eval_names, eval_args)
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
                cmd=self.get_command('agent', args)
            )

    def kill_agents(self, agent_names):
        assert self.is_launched('agent'), 'agents not yet launched'
        for name in agent_names:
            self._tmux.kill(
                session_name=self.agent_session,
                window_name=str(name)
            )

    def add_evals(self, eval_names, eval_args):
        eval_names, eval_args = self._get_agent_info(
            eval_names, eval_args
        )
        # should not duplicate agent name
        assert not (set(self.get_running_evals()) & set(eval_names)), \
            'some evaluators already running, cannot launch duplicates.'
        for eval_name, args in zip(eval_names, eval_args):
            self._tmux.run(
                session_name=self.learner_session,
                window_name=eval_name,
                cmd=self.get_command('eval', args)
            )

    def kill_evals(self, eval_names):
        assert self.is_launched('learner'), 'evaluators not yet launched'
        for name in eval_names:
            self._tmux.kill(
                session_name=self.learner_session,
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
            err = self._tmux.check_error(
                session_name=sess,
                window_name=win,
                history=200,
                after_context=1
            )
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



class TmuxClusterOld(object):
    """
    Launch the following in order:
    1. Loggerplex (distributed logging server script)
    2. Tensorplex (distributed tensorplex server script)
    3. Tensorboard, `tensorboard --logdir . --port <tensorboard_port>`
    4. Parameter server (standalone script)
    5. Replay server
    6. Learner
    7. Evaluator (=None to skip evaluation)
    8. Army of agents
    """
    LOGGERPLEX_SCRIPT = 'surreal.session.run_loggerplex_server'
    TENSORPLEX_SCRIPT = 'surreal.session.run_tensorplex_server'
    PS_SCRIPT = 'surreal.session.run_parameter_server'

    def __init__(self,
                 cluster_name,
                 session_config,
                 agent_script,
                 learner_script,
                 replay_script,
                 eval_script,
                 start_dir='.',
                 preamble_cmd=None,
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
        self.replay_cmd = self._get_python_cmd(replay_script)
        if eval_script is None:
            self.eval_cmd = None
        else:
            self.eval_cmd = self._get_python_cmd(eval_script)
        self.infras_session = 'infras-' + cluster_name
        self.agent_session = 'agent-' + cluster_name
        self.learner_session = 'learner-' + cluster_name
        self._tmux = TmuxRunner(
            start_dir=start_dir,
            preamble_cmd=preamble_cmd,
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

    def get_running_evals(self):
        return [win for win
                in self._tmux.list_window_names(self.learner_session)
                if not win.startswith('learner')]

    def _get_cmd_with_json(self, script):
        script = self._get_python_cmd(script)
        # dump config to JSON as command line arg
        return script + ' ' + shlex.quote(json.dumps(self.config))

    def launch(self,
               agent_names,
               agent_args,
               eval_names=None,
               eval_args=None):
        # Infrastructure session
        if not self.is_launched('infras'):
            self._tmux.run(
                session_name=self.infras_session,
                window_name='loggerplex',
                cmd=self._get_cmd_with_json(self.LOGGERPLEX_SCRIPT)
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='tensorplex',
                cmd=self._get_cmd_with_json(self.TENSORPLEX_SCRIPT)
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='tensorboard',
                cmd='tensorboard --logdir {} --port {}'.format(
                    self.config.folder,
                    self.config.tensorplex.tensorboard_port
                )
            )
            self._tmux.run(
                session_name=self.infras_session,
                window_name='ps',
                cmd=self._get_cmd_with_json(self.PS_SCRIPT)
            )
        # Learner session
        if not self.is_launched('learner'):
            self._tmux.run(
                session_name=self.learner_session,
                window_name='replay',
                cmd=self.replay_cmd
            )
            self._tmux.run(
                session_name=self.learner_session,
                window_name='learner',
                cmd=self.learner_cmd
            )
            if self.eval_cmd is not None:
                self.add_evals(eval_names, eval_args)
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

    def add_evals(self, eval_names, eval_args):
        eval_names, eval_args = self._get_agent_info(
            eval_names, eval_args
        )
        # should not duplicate agent name
        assert not (set(self.get_running_evals()) & set(eval_names)), \
            'some evaluators already running, cannot launch duplicates.'
        for eval_name, args in zip(eval_names, eval_args):
            self._tmux.run(
                session_name=self.learner_session,
                window_name=eval_name,
                cmd=self.eval_cmd + ' ' + args
            )

    def kill_evals(self, eval_names):
        assert self.is_launched('learner'), 'evaluators not yet launched'
        for name in eval_names:
            self._tmux.kill(
                session_name=self.learner_session,
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
            err = self._tmux.check_error(
                session_name=sess,
                window_name=win,
                history=200,
                after_context=1
            )
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

