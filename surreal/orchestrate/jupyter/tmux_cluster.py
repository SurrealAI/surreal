"""
Interactive cluster control in a remote Jupyter notebook.
To set up remote port forwarding for Jupyter and tensorboard:
http://amber-md.github.io/pytraj/latest/tutorials/remote_jupyter_notebook
"""
from .interactive_util import *
from surreal.orchestrate import TmuxCluster


set_cluster_var, _get_global_var = create_interactive_suite(
    suite_name='cluster',
    var_class=TmuxCluster,
)


def numbered_agents(n1, n2=None):
    """
    Generate default agent name and args for the cluster.
    Default names are "A<n>": A12, A3, etc.
    Default args are simply "<n>"

    Two ways to call this function:
    1. numbered_agents([list of agent indices])
    2. numbered_agents(n1, n2) -> range(n1, n2)
    """
    if n2 is None:
        U.assert_type(n1, list)
        indices = n1
    else:
        U.assert_type(n1, int)
        U.assert_type(n2, int)
        indices = range(n1, n2)
    agent_names = ['A' + str(i) for i in indices]
    agent_args = [str(i) for i in indices]
    return {
        'agent_names': agent_names,
        'agent_args': agent_args
    }


def numbered_evals(n):
    """
    Command line args spec: please see `surreal.main.run_cartpole_eval`
    """
    spec = {
        'eval_names': ['eval_d'],  # tmux window name
        'eval_args': ['0 --mode eval_deterministic']
    }
    for i in range(n):
        spec['eval_names'].append('eval_s-{}'.format(i))
        spec['eval_args'].append(['--mode eval_stochastic', i])
    return spec


def ls():
    return _get_global_var().list_windows()


def launch_(agent_names, agent_args, eval_names=None, eval_args=None):
    _get_global_var().launch(
        agent_names=agent_names,
        agent_args=agent_args,
        eval_names=eval_names,
        eval_args=eval_args
    )


def launch(n1, n2=None, *, eval=1):
    _get_global_var().launch(
        **numbered_agents(n1, n2),
        **numbered_evals(eval),
    )


def add_agents_(agent_names, agent_args):
    _get_global_var().add_agents(
        agent_names=agent_names,
        agent_args=agent_args
    )


def add_agents(n1, n2=None):
    _get_global_var().add_agents(**numbered_agents(n1, n2))


def kill_(agent_names):
    _get_global_var().kill_agents(agent_names)


def kill(n1, n2=None):
    _get_global_var().kill_agents(numbered_agents(n1, n2)['agent_names'])


def killall(clean=True):
    """
    Args:
        clean: if True, removes the experiment dir.
            Reads from `TmuxCluster.config.folder`
    """
    cluster = _get_global_var()
    cluster.killall()
    if clean:
        # protection
        folder = cluster.config.folder
        assert folder.strip('/') not in ['~', '.', '']
        U.f_remove(folder)
        print('Experiment folder "{}" removed.'.format(folder))


def error(group=None, window=None):
    _get_global_var().print_error(group, window)


def stdout(group=None, window=None, history=0):
    _get_global_var().print_stdout(group, window, history=history)