

class CommandGenerator:
    def __init__(self, *,
                 num_agents,
                 num_evals,
                 executable,
                 config_commands=None,
                 ignore_python='auto'):
        """
        Args:
            num_agents: number of agents to run
            num_evals: number of evals to run
            executable: path to .py executable in the pod
            config_commands (arr): additional commands that pass
                to user-defined config.py, after "--"
            ignore_python (Ture, False, auto):
                when True, omit ["python", -u"],
                'auto': True when executable does not end with '.py'
                useful for using excecutables
        """
        self.num_agents = num_agents
        self.num_evals = num_evals
        self.executable = executable
        self.config_commands = config_commands
        if ignore_python == 'auto':
            ignore_python = executable[-3:] != '.py'
        self.ignore_python = ignore_python

    def get_command(self, role):
        command = []
        if not self.ignore_python:
            command += ['python', '-u']
        command += [self.executable]
        command += [role]
        command += ['--']
        if self.config_commands is not None:
            command += self.config_commands
        return ' '.join(command)
