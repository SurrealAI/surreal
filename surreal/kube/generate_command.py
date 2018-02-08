import shlex

class CommandGenerator():
    def __init__(self, config_path, config_command=None, service_url=None):
        self.config_path = config_path
        self.config_command = config_command
        self.service_url = service_url

    def get_command(self, role, args=None):
        if args is None:
            args = []
        command = ['python -u -m', 'surreal.main_scripts.runner', self.config_path]
        if self.service_url is not None:
            command += ['--service-url', shlex.quote(self.service_url)]
        command += [role]
        command += args
        if self.config_command is not None:
            command += ['--', self.config_command]
        return ' '.join(command)

    def generate(self, num_agents):
        di = {}
        for role in ['tensorplex', 'tensorboard', 'loggerplex', 'ps', 'replay', 'learner']:
            di[role] = self.get_command(role)
        di['agent'] = [self.get_command('agent', [str(i)]) for i in range(num_agents)]
        di['eval'] = [self.get_command('eval', ['--mode', 'deterministic', '--id', '0'])]
        return di

"""
    generator = CommandGenerator(*args)
    di = 
"""
if __name__ == "__main__":
    gen = CommandGenerator('~/.abc',
                           config_command='--test-command def',
                           service_url='foo.bar.com')
    di = gen.generate(3)
    print(di)