import os
import libtmux
from libtmux.exc import LibTmuxException
import pprint
import re
import time

# matches "ZeroDivisionError: divided by zero"
_exception_re = re.compile('([a-zA-Z_0-9\.]+: .+)|'
                           '([a-zA-Z]+(Error|Exception).*)')


class TmuxRunner(object):
    def __init__(self,
                 start_dir=None,
                 preamble_cmd=None,
                 verbose=True,
                 dry_run=False):
        """

        Args:
            start_dir:
            preamble_cmd: execute before every run(), can be overriden in run arg
                can be either a string or a list of strings to execute in order
            verbose:
            dry_run:
        """
        self._server = libtmux.Server()
        self._start_dir = os.path.expanduser(start_dir)
        self._verbose = verbose
        self._dry_run = dry_run
        self._preamble_cmd = preamble_cmd
        # two-level dict of session:window:cmd
        self.records = {}
        if self._dry_run:
            print('TmuxExecutor: dry run.')

    def get_session(self, session_name):
        try:
            return self._server.find_where({'session_name': session_name})
        except LibTmuxException:
            return None

    def get_window(self, session, window_name):
        if isinstance(session, str):
            # get actual session from session name
            session = self.get_session(session)
            if session is None:
                return None
        try:
            return session.find_where({'window_name': window_name})
        except LibTmuxException:
            return None

    def run(self, session_name, window_name, cmd,
            preamble_cmd=None, start_dir=None):
        self._vprint('{}:{}\t>>\t{}'.format(session_name, window_name, cmd))
        if self._dry_run:
            return
        if not preamble_cmd:
            preamble_cmd = self._preamble_cmd

        if start_dir is None:
            start_dir = self._start_dir
        else:
            start_dir = os.path.expanduser(start_dir)

        session = self.get_session(session_name)
        if session is None:
            session = self._server.new_session(
                session_name,
                start_directory=start_dir
            )
            window = session.attached_window
            window.rename_window(window_name)
        else:
            window = self.get_window(session, window_name)
            if window is None:
                window = session.new_window(window_name,
                                            start_directory=start_dir)
        pane = window.attached_pane
        if preamble_cmd:
            if isinstance(preamble_cmd, list):
                for p in preamble_cmd:
                    pane.send_keys(p)
            else:
                pane.send_keys(preamble_cmd)
        pane.send_keys(cmd)
        # add session/window/cmd info to records
        if session_name in self.records:
            self.records[session_name][window_name] = cmd
        else:
            self.records[session_name] = {window_name: cmd}

    def kill(self, session_name, window_name=None):
        assert not self._dry_run
        session = self.get_session(session_name)
        if session:
            if window_name is None:
                # kill entire session
                session.kill_session()
                self._vprint('session', session_name, 'killed')
            else:
                window = self.get_window(session, window_name)
                if window is None: return
                window.kill_window()
                self._vprint('{}:{} killed'.format(session_name, window_name))

    def killall(self):
        for session in self.records:
            self.kill(session)

    def list_window_names(self, session):
        if isinstance(session, str):
            session = self.get_session(session)
        if session is None:
            return []
        return [w.name for w in session.windows]

    def list_session_names(self):
        """
        Warnings:
            May include sessions that are not created by this TmuxRunner!
        """
        try:
            return [s.name for s in self._server.sessions]
        except LibTmuxException:
            return []

    def get_stdout(self, session_name, window_name, history=0):
        """
        Args:
            history: number of lines before the visible pane to be captured.
        """
        window = self.get_window(session_name, window_name)
        if window is None:
            raise ValueError('window "{}" does not exist'.format(window_name))
        pane = window.attached_pane
        cmd = ['capture-pane', '-p']
        if history != 0:
            cmd += ['-S', str(-abs(history))]
        return pane.cmd(*cmd).stdout

    def check_error(self, session_name, window_name,
                    history=0, before_context=0, after_context=0):
        """
        Implements a very crude and not very reliable way to detect exception.

        Capture-pane manual:
        https://docs.oracle.com/cd/E86824_01/html/E54763/tmux-1.html

        Returns:
            str: if the *visible* pane shows "Traceback (most recent call last):"
              return the
            False: if the *visible* pane does not show Traceback
            None: if the window doesn't exist
        """
        stdout = self.get_stdout(session_name, window_name, history=history)
        errlines = []
        i = 0
        while i < len(stdout):
            line = stdout[i]
            if line.startswith('Traceback (most recent call last)'):
                for j in range(i+1, len(stdout)):
                    if _exception_re.match(stdout[j]):
                        start = max(0, i-before_context)
                        end = j + 1 + after_context
                        errlines.extend(stdout[start:end])
                        i = j
                        break
            i += 1
        if errlines:
            return '\n'.join(errlines)
        else:
            return None

    def print_records(self):
        pprint.pprint(self.records, indent=4)

    def _vprint(self, *args):
        if self._verbose:
            print(*args)
