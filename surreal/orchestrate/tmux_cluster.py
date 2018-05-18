import itertools
from surreal.kube.generate_command import CommandGenerator
from symphony.engine import Cluster

# import time
# import json
# import shlex

# from collections import OrderedDict
# from surreal.session.config import Config
# from surreal.session.default_configs import BASE_SESSION_CONFIG
# import surreal.utils as U
# from surreal.main_scripts.runner import load_config

# from .tmux_runner import TmuxRunner


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
                 experiment_name,
                 num_agents,
                 config_path,
                 experiment_folder,
                 start_dir='.',
                 preamble_cmds=None,
                 config_command=None,
                ):
        """
        Args:
            TODO: Add more documentation for config files.
            TODO: docs need updates
            @config_path: File system path to a .py config file 
            For now look at the example of surreal/main/ddpg_configs.py
            @start_dir: Tmux initial directory
            @preamble_cmd: Commands to run in a tmux window before running the surreal process
            E.g. source activate [name of your virtual env]
            @config_command: Command to supply to the config file through the runner's --config-command argument
            Will be escaped by shlex.quote
            @dry_run: Set for tmux
        """
        self.cluster = Cluster.new('tmux')
        self.preamble_cmds = preamble_cmds
        self.start_dir = start_dir
        self.experiment_name = experiment_name
        self.num_agents = num_agents
        self.num_evals = 1
        self.experiment_folder = experiment_folder
        self.config_command = config_command
        self.generator = CommandGenerator(
            num_agents=num_agents,
            experiment_folder=experiment_folder,
            config_py=config_path,
            config_command=config_command,
            service_url=None
        )
        self.cmd_dict = self.generator.generate()
        self.exp = None

    def launch(self, dry_run=False):
        if self.exp:
            self.cluster.delete(self.exp.name)
        cmd_dict = self.cmd_dict
        exp = self.cluster.new_experiment(self.experiment_name,
                                          preamble_cmds=self.preamble_cmds,
                                          start_dir=self.start_dir,
                                          port_range=list(range(9000,10000)))
        self.exp = exp
        learner = exp.new_process('learner', cmds=[cmd_dict['learner']])
        replay = exp.new_process('replay', cmds=[cmd_dict['replay']])
        ps = exp.new_process('ps', cmds=[cmd_dict['ps']])
        tensorboard = exp.new_process('tensorboard', cmds=[cmd_dict['tensorboard']])
        tensorplex = exp.new_process('tensorplex', cmds=[cmd_dict['tensorplex']])
        loggerplex = exp.new_process('loggerplex', cmds=[cmd_dict['loggerplex']])

        agents = []
        for i, arg in enumerate(cmd_dict['agent']):
            agent_p = exp.new_process('agent-{}'.format(i), cmds=[arg])
            agents.append(agent_p)

        evals = []
        self.num_evals = len(cmd_dict['eval'])
        for i, arg in enumerate(cmd_dict['eval']):
            eval_p = exp.new_process('eval-{}'.format(i), cmds=[arg])
            evals.append(eval_p)

        for proc in itertools.chain(agents, evals):
            proc.connects('ps-frontend')
            proc.connects('collector-frontend')

        ps.binds('ps-frontend')
        ps.binds('ps-backend')
        ps.connects('parameter-publish')

        replay.binds('collector-frontend')
        replay.binds('sampler-frontend')
        replay.binds('collector-backend')
        replay.binds('sampler-backend')

        learner.connects('sampler-frontend')
        learner.binds('parameter-publish')
        learner.binds('prefetch-queue')

        tensorplex.binds('tensorplex')
        loggerplex.binds('loggerplex')

        for proc in itertools.chain(agents, evals, [ps, replay, learner]):
            proc.connects('tensorplex')
            proc.connects('loggerplex')

        tensorboard.exposes({'tensorboard': 6006})

        self.cluster.launch(exp, dry_run=dry_run)

    def killall(self):
        try:
            self.cluster.delete(self.experiment_name)
        except ValueError as e:
            print(e)

    def print_all(self, tail=100):
        self.print_agents(tail)
        self.print_learner(tail)
        self.print_infras(tail)

    def print_agents(self, tail=100):
        components = ['agent-{}'.format(i) for i in range(self.num_agents)]
        self.print_header('agent')
        self.print_stdout(components, tail=tail)

    def print_learner(self, tail=100):
        components = ['learner']
        components += ['eval-{}'.format(i) for i in range(self.num_evals)]
        components += ['replay']
        self.print_header('learner')
        self.print_stdout(components, tail=tail)

    def print_infras(self, tail=100):
        components = ['ps', 'tensorplex', 'tensorboard', 'loggerplex']
        self.print_header('infras')
        self.print_stdout(components, tail=tail)

    def print_header(self, name, sep='='):
        """
        sep: separator symbol "=" in "====== <win name> ======"
        """
        print(sep * 20, name, sep * 20)

    def print_stdout(self, components, tail=100):
        """
        Args:
            tail: the last lines to print
        """
        for component in components:
            output = self.cluster.get_log(self.exp.name, component, tail=tail)
            print('\n'.join(output))

    # def add_agents(self, agent_names, agent_args):
    #     agent_names, agent_args = self._get_agent_info(
    #         agent_names, agent_args
    #     )
    #     # should not duplicate agent name
    #     assert not (set(self.get_running_agents()) & set(agent_names)), \
    #         'some agents already running, cannot launch duplicates.'
    #     for agent_name, args in zip(agent_names, agent_args):
    #         self._tmux.run(
    #             session_name=self.agent_session,
    #             window_name=agent_name,
    #             cmd=self.get_command('agent', args)
    #         )

    # def kill_agents(self, agent_names):
    #     assert self.is_launched('agent'), 'agents not yet launched'
    #     for name in agent_names:
    #         self._tmux.kill(
    #             session_name=self.agent_session,
    #             window_name=str(name)
    #         )

    # def add_evals(self, eval_names, eval_args):
    #     eval_names, eval_args = self._get_agent_info(
    #         eval_names, eval_args
    #     )
    #     # should not duplicate agent name
    #     assert not (set(self.get_running_evals()) & set(eval_names)), \
    #         'some evaluators already running, cannot launch duplicates.'
    #     for eval_name, args in zip(eval_names, eval_args):
    #         self._tmux.run(
    #             session_name=self.learner_session,
    #             window_name=eval_name,
    #             cmd=self.get_command('eval', args)
    #         )

    # def kill_evals(self, eval_names):
    #     assert self.is_launched('learner'), 'evaluators not yet launched'
    #     for name in eval_names:
    #         self._tmux.kill(
    #             session_name=self.learner_session,
    #             window_name=str(name)
    #         )

    # def _iterate_all_windows(self):
    #     for sess in [self.agent_session,
    #                  self.learner_session,
    #                  self.infras_session]:
    #         for win in self._tmux.list_window_names(sess):
    #             yield sess, win

    # def _iterated_filtered_windows(self, group=None, window=None):
    #     if group is None:
    #         assert window is None, \
    #             'when group is specified, window should be left as None'
    #         yield from self._iterate_all_windows()
    #     else:
    #         sess = self._session_group(group)
    #         if window is None:
    #             for win in self._tmux.list_window_names(sess):
    #                 yield sess, win
    #         else:
    #             window = str(window)
    #             yield sess, window  # just one iteration

    # def list_windows(self):
    #     windict = OrderedDict()
    #     for sess, win in self._iterate_all_windows():
    #         if sess in windict:
    #             windict[sess].append(win)
    #         else:
    #             windict[sess] = [win]
    #     return windict


    # def check_error(self, group=None, window=None):
    #     """
    #     For group and window semantics, refer to `get_stdout`

    #     Returns:
    #         OrderedDict({"session:window": "error-message"})
    #     """
    #     outdict = OrderedDict()
    #     for sess, win in self._iterated_filtered_windows(group, window):
    #         err = self._tmux.check_error(
    #             session_name=sess,
    #             window_name=win,
    #             history=200,
    #             after_context=1
    #         )
    #         if err:
    #             outdict['{}:{}'.format(sess, win)] = err
    #     return outdict

    # def print_error(self, group=None, window=None, sep='='):
    #     errdict = self.check_error(group, window)
    #     if len(errdict) == 0:
    #         if group is None and window is None:
    #             print('No error found in cluster.')
    #         elif window is None:
    #             print('No error found in group "{}" (tmux session "{}").'
    #                   .format(group, self._session_group(group)))
    #         else:
    #             print('No error found in "{}:{}" window'
    #                   .format(self._session_group(group), window))
    #     else:
    #         for win, out in errdict.items():
    #             print(sep*20, win, sep*20)
    #             print(out)


    # def get_command(self, mode, args=None):
    #     """
    #         mode is agent/learner/...
    #         args is the surreal defined argument to give to agent/learner, in a string!!!!
    #     """
    #     command = ['python -u -m', 'surreal.main_scripts.runner', self.config_path]
    #     command += ['--experiment-folder', self.experiment_folder]
    #     command += [mode]
    #     if args is not None:
    #         command += [args]
    #     if self.config_command is not None:
    #         command += ['--', self.config_command]
    #     return ' '.join(command)

    # def _get_agent_info(self, agent_names, agent_args_):
    #     U.assert_type(agent_names, list)
    #     U.assert_type(agent_args_, list)
    #     agent_names = [str(_name) for _name in agent_names]
    #     assert len(agent_names) == len(set(agent_names)), \
    #         'must not duplicate agent names'
    #     assert len(agent_names) == len(agent_args_)
    #     agent_args = []
    #     for cmd_args in agent_args_:
    #         if cmd_args is None:
    #             cmd = ''
    #         elif isinstance(cmd_args, str):
    #             cmd = cmd_args
    #         elif isinstance(cmd_args, list):
    #             cmd = ' '.join(str(x) for x in cmd_args)
    #         else:
    #             raise ValueError('Must be a list of command line arg list '
    #                              'OR a list of command strings.')
    #         agent_args.append(cmd)
    #     return agent_names, agent_args

    # def _session_group(self, group):
    #     assert group in ['agent', 'learner', 'infras']
    #     return {
    #         'agent': self.agent_session,
    #         'learner': self.learner_session,
    #         'infras': self.infras_session
    #     }[group]

    # def is_launched(self, group):
    #     return self._session_group(group) in self._tmux.list_session_names()

    # def get_running_agents(self):
    #     return self._tmux.list_window_names(self.agent_session)

    # def get_running_evals(self):
    #     return [win for win
    #             in self._tmux.list_window_names(self.learner_session)
    #             if not win.startswith('learner')]

    # def _get_cmd_with_json(self, script):
    #     script = self._get_python_cmd(script)
    #     # dump config to JSON as command line arg
    #     return script + ' ' + shlex.quote(json.dumps(self.config))

# class TmuxClusterOld(object):
#     """
#     Launch the following in order:
#     1. Loggerplex (distributed logging server script)
#     2. Tensorplex (distributed tensorplex server script)
#     3. Tensorboard, `tensorboard --logdir . --port <tensorboard_port>`
#     4. Parameter server (standalone script)
#     5. Replay server
#     6. Learner
#     7. Evaluator (=None to skip evaluation)
#     8. Army of agents
#     """
#     LOGGERPLEX_SCRIPT = 'surreal.session.run_loggerplex_server'
#     TENSORPLEX_SCRIPT = 'surreal.session.run_tensorplex_server'
#     PS_SCRIPT = 'surreal.session.run_parameter_server'

#     def __init__(self,
#                  cluster_name,
#                  session_config,
#                  agent_script,
#                  learner_script,
#                  replay_script,
#                  eval_script,
#                  start_dir='.',
#                  preamble_cmd=None,
#                  dry_run=False
#                  ):
#         """
#         Args:
#             session_config:
#             agent_args: list of list of command line args OR command strings.
#                 Each agent might have a different command line invocation,
#                 such as different names and exploration strategies. E.g.
#                 [
#                     ['--explore', 'strategy1', '--id', 10],
#                     ['--explore', 'strategy2', '--id', 13, '--anneal', 0.5],
#                     '--explore strat3 --id 22'  # or simply a long string
#                 ]
#         """
#         self.config = Config(session_config).extend(BASE_SESSION_CONFIG)
#         self.agent_cmd = self._get_python_cmd(agent_script)
#         self.learner_cmd = self._get_python_cmd(learner_script)
#         self.replay_cmd = self._get_python_cmd(replay_script)
#         if eval_script is None:
#             self.eval_cmd = None
#         else:
#             self.eval_cmd = self._get_python_cmd(eval_script)
#         self.infras_session = 'infras-' + cluster_name
#         self.agent_session = 'agent-' + cluster_name
#         self.learner_session = 'learner-' + cluster_name
#         self._tmux = TmuxRunner(
#             start_dir=start_dir,
#             preamble_cmd=preamble_cmd,
#             verbose=True,
#             dry_run=dry_run
#         )

#     def _get_python_cmd(self, python_script):
#         if python_script.startswith('python'):
#             return python_script  # already a command
#         if not python_script.endswith('.py') and '/' in python_script:
#             raise ValueError('Ill-formed python script ' + python_script +
#                              ' should be either pkg1.pkg2.myscript or '
#                              'pkg1/pkg2/myscript.py')
#         if python_script.endswith('.py'):
#             return 'python -u ' + python_script
#         else:
#             # python -m surreal.main.run_cartpole
#             return 'python -u -m ' + python_script

#     def _get_agent_info(self, agent_names, agent_args_):
#         U.assert_type(agent_names, list)
#         U.assert_type(agent_args_, list)
#         agent_names = [str(_name) for _name in agent_names]
#         assert len(agent_names) == len(set(agent_names)), \
#             'must not duplicate agent names'
#         assert len(agent_names) == len(agent_args_)
#         agent_args = []
#         for cmd_args in agent_args_:
#             if cmd_args is None:
#                 cmd = ''
#             elif isinstance(cmd_args, str):
#                 cmd = cmd_args
#             elif isinstance(cmd_args, list):
#                 cmd = ' '.join(str(x) for x in cmd_args)
#             else:
#                 raise ValueError('Must be a list of command line arg list '
#                                  'OR a list of command strings.')
#             agent_args.append(cmd)
#         return agent_names, agent_args

#     def _session_group(self, group):
#         assert group in ['agent', 'learner', 'infras']
#         return {
#             'agent': self.agent_session,
#             'learner': self.learner_session,
#             'infras': self.infras_session
#         }[group]

#     def is_launched(self, group):
#         return self._session_group(group) in self._tmux.list_session_names()

#     def get_running_agents(self):
#         return self._tmux.list_window_names(self.agent_session)

#     def get_running_evals(self):
#         return [win for win
#                 in self._tmux.list_window_names(self.learner_session)
#                 if not win.startswith('learner')]

#     def _get_cmd_with_json(self, script):
#         script = self._get_python_cmd(script)
#         # dump config to JSON as command line arg
#         return script + ' ' + shlex.quote(json.dumps(self.config))

#     def launch(self,
#                agent_names,
#                agent_args,
#                eval_names=None,
#                eval_args=None):
#         # Infrastructure session
#         if not self.is_launched('infras'):
#             self._tmux.run(
#                 session_name=self.infras_session,
#                 window_name='loggerplex',
#                 cmd=self._get_cmd_with_json(self.LOGGERPLEX_SCRIPT)
#             )
#             self._tmux.run(
#                 session_name=self.infras_session,
#                 window_name='tensorplex',
#                 cmd=self._get_cmd_with_json(self.TENSORPLEX_SCRIPT)
#             )
#             self._tmux.run(
#                 session_name=self.infras_session,
#                 window_name='tensorboard',
#                 cmd='tensorboard --logdir {} --port {}'.format(
#                     self.config.folder,
#                     self.config.tensorplex.tensorboard_port
#                 )
#             )
#             self._tmux.run(
#                 session_name=self.infras_session,
#                 window_name='ps',
#                 cmd=self._get_cmd_with_json(self.PS_SCRIPT)
#             )
#         # Learner session
#         if not self.is_launched('learner'):
#             self._tmux.run(
#                 session_name=self.learner_session,
#                 window_name='replay',
#                 cmd=self.replay_cmd
#             )
#             self._tmux.run(
#                 session_name=self.learner_session,
#                 window_name='learner',
#                 cmd=self.learner_cmd
#             )
#             if self.eval_cmd is not None:
#                 self.add_evals(eval_names, eval_args)
#         # Agent session
#         if not self.is_launched('agent'):
#             self.add_agents(agent_names, agent_args)

#     def add_agents(self, agent_names, agent_args):
#         agent_names, agent_args = self._get_agent_info(
#             agent_names, agent_args
#         )
#         # should not duplicate agent name
#         assert not (set(self.get_running_agents()) & set(agent_names)), \
#             'some agents already running, cannot launch duplicates.'
#         for agent_name, args in zip(agent_names, agent_args):
#             self._tmux.run(
#                 session_name=self.agent_session,
#                 window_name=agent_name,
#                 cmd=self.agent_cmd + ' ' + args
#             )

#     def kill_agents(self, agent_names):
#         assert self.is_launched('agent'), 'agents not yet launched'
#         for name in agent_names:
#             self._tmux.kill(
#                 session_name=self.agent_session,
#                 window_name=str(name)
#             )

#     def add_evals(self, eval_names, eval_args):
#         eval_names, eval_args = self._get_agent_info(
#             eval_names, eval_args
#         )
#         # should not duplicate agent name
#         assert not (set(self.get_running_evals()) & set(eval_names)), \
#             'some evaluators already running, cannot launch duplicates.'
#         for eval_name, args in zip(eval_names, eval_args):
#             self._tmux.run(
#                 session_name=self.learner_session,
#                 window_name=eval_name,
#                 cmd=self.eval_cmd + ' ' + args
#             )

#     def kill_evals(self, eval_names):
#         assert self.is_launched('learner'), 'evaluators not yet launched'
#         for name in eval_names:
#             self._tmux.kill(
#                 session_name=self.learner_session,
#                 window_name=str(name)
#             )

#     def _iterate_all_windows(self):
#         for sess in [self.agent_session,
#                      self.learner_session,
#                      self.infras_session]:
#             for win in self._tmux.list_window_names(sess):
#                 yield sess, win

#     def _iterated_filtered_windows(self, group=None, window=None):
#         if group is None:
#             assert window is None, \
#                 'when group is specified, window should be left as None'
#             yield from self._iterate_all_windows()
#         else:
#             sess = self._session_group(group)
#             if window is None:
#                 for win in self._tmux.list_window_names(sess):
#                     yield sess, win
#             else:
#                 window = str(window)
#                 yield sess, window  # just one iteration

#     def list_windows(self):
#         windict = OrderedDict()
#         for sess, win in self._iterate_all_windows():
#             if sess in windict:
#                 windict[sess].append(win)
#             else:
#                 windict[sess] = [win]
#         return windict

#     def get_stdout(self, group=None, window=None, history=0):
#         """
#         Args:
#             group: [agent, learner, infras] None for all
#             window: get specific window. None for all windows.
#                 If group is None, window must also be None.
#         Returns:
#             OrderedDict({"session:window": "pane stdout"})
#             pane stdout captures only the visible part unless you specify history
#         """
#         outdict = OrderedDict()
#         for sess, win in self._iterated_filtered_windows(group, window):
#             stdout = self._tmux.get_stdout(sess, win, history=history)
#             U.assert_type(stdout, list)  # internal
#             outdict['{}:{}'.format(sess, win)] = '\n'.join(stdout)
#         return outdict

#     def print_stdout(self, group=None, window=None, history=0, sep='='):
#         """
#         Args:
#             group:
#             window:
#             history:
#             sep: separator symbol "=" in "====== <win name> ======"
#         """
#         for win, out in self.get_stdout(
#                 group=group,
#                 window=window,
#                 history=history
#         ).items():
#             print(sep*20, win, sep*20)
#             print(out)

#     def check_error(self, group=None, window=None):
#         """
#         For group and window semantics, refer to `get_stdout`

#         Returns:
#             OrderedDict({"session:window": "error-message"})
#         """
#         outdict = OrderedDict()
#         for sess, win in self._iterated_filtered_windows(group, window):
#             err = self._tmux.check_error(
#                 session_name=sess,
#                 window_name=win,
#                 history=200,
#                 after_context=1
#             )
#             if err:
#                 outdict['{}:{}'.format(sess, win)] = err
#         return outdict

#     def print_error(self, group=None, window=None, sep='='):
#         errdict = self.check_error(group, window)
#         if len(errdict) == 0:
#             if group is None and window is None:
#                 print('No error found in cluster.')
#             elif window is None:
#                 print('No error found in group "{}" (tmux session "{}").'
#                       .format(group, self._session_group(group)))
#             else:
#                 print('No error found in "{}:{}" window'
#                       .format(self._session_group(group), window))
#         else:
#             for win, out in errdict.items():
#                 print(sep*20, win, sep*20)
#                 print(out)

#     def killall(self):
#         self._tmux.kill(self.agent_session)
#         self._tmux.kill(self.learner_session)
#         self._tmux.kill(self.infras_session)

