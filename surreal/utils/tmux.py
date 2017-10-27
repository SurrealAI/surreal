import os
import pprint
import libtmux
from libtmux.exc import LibTmuxException


def tmux_get_session(server, session_name):
    try:
        return server.find_where({'session_name': session_name})
    except LibTmuxException:
        return None


def tmux_get_window(session, window_name):
    return session.find_where({'window_name': window_name})


class TmuxExecutor:
    def __init__(self, start_dir='.', verbose=True, dry_run=False):
        self.server = libtmux.Server()
        self.start_dir = os.path.expanduser(start_dir)
        self.verbose = verbose
        self.dry_run = dry_run
        # two-level dict of session:window:cmd
        self.records = {}
        if self.dry_run:
            print('TmuxExecutor: dry run.')

    def run(self, session_name, window_name, cmd, start_dir=None):
        if self.verbose:
            print('{}:{}\t>>\t{}'.format(session_name, window_name, cmd))
        if self.dry_run:
            return

        if start_dir is None:
            start_dir = self.start_dir
        else:
            start_dir = os.path.expanduser(start_dir)

        session = tmux_get_session(self.server, session_name)
        if session is None:
            session = self.server.new_session(session_name,
                                              start_directory=start_dir)
            window = session.attached_window
            window.rename_window(window_name)
        else:
            window = tmux_get_window(session, window_name)
            if window is None:
                window = session.new_window(window_name,
                                            start_directory=start_dir)
        pane = window.attached_pane
        pane.send_keys(cmd)
        # add session/window/cmd info to records
        if session_name in self.records:
            self.records[session_name][window_name] = cmd
        else:
            self.records[session_name] = {window_name: cmd}

    def print_records(self):
        pprint.pprint(self.records, indent=4)
