"""
Interactive cluster control in a remote Jupyter notebook.
To set up remote port forwarding for Jupyter and tensorboard:
http://amber-md.github.io/pytraj/latest/tutorials/remote_jupyter_notebook
"""
# convenient imports for initializing Jupyter
import os, sys, json, time, re, random
from collections import *
from time import sleep
import numpy as np
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn
import math
import surreal.utils as U
from surreal.session import TmuxCluster

pp = U.pprint


_CLUSTER_GLOBALS = {
    'varname': 'cluster',
    'globals': None  # global dict
}


def _check_globals(varname, global_dict):
    assert varname in global_dict, \
        'Variable "{}" does not exist in global scope.'.format(varname)


def set_cluster_var(varname, global_dict):
    """
    Put your cluster variable name here.
    All other commands will get the cluster instance from globals()['varname']

    Args:
        varname: string of cluster variable name
        global_dict: pass in `globals()`
    """
    global _CLUSTER_GLOBALS
    U.assert_type(varname, str)
    U.assert_type(global_dict, dict)
    _check_globals(varname, global_dict)
    _CLUSTER_GLOBALS['varname'] = varname
    _CLUSTER_GLOBALS['globals'] = global_dict


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


def _get_global_cluster():
    global _CLUSTER_GLOBALS
    if not _CLUSTER_GLOBALS['globals']:
        raise RuntimeError('Please call set_cluster_var("clustervar", globals()) first.')
    varname = _CLUSTER_GLOBALS['varname']
    global_dict = _CLUSTER_GLOBALS['globals']
    _check_globals(varname, global_dict)
    cluster = global_dict[varname]
    U.assert_type(cluster, TmuxCluster), \
        'Variable "{}" must be a TmuxCluster instance'.format(varname)
    return cluster


def ls():
    return _get_global_cluster().list_windows()


def launch_(agent_names, agent_args):
    _get_global_cluster().launch(
        agent_names=agent_names,
        agent_args=agent_args
    )


def launch(n1, n2=None):
    _get_global_cluster().launch(**numbered_agents(n1, n2))


def add_(agent_names, agent_args):
    _get_global_cluster().add_agents(
        agent_names=agent_names,
        agent_args=agent_args
    )


def add(n1, n2=None):
    _get_global_cluster().add_agents(**numbered_agents(n1, n2))


def kill_(agent_names):
    _get_global_cluster().kill_agents(agent_names)


def kill(n1, n2=None):
    _get_global_cluster().kill_agents(numbered_agents(n1, n2)['agent_names'])


def killall(clean=True):
    """
    Args:
        clean: if True, removes the experiment dir.
            Reads from `TmuxCluster.config.folder`
    """
    cluster = _get_global_cluster()
    cluster.killall()
    if clean:
        # protection
        folder = cluster.config.folder
        assert folder.strip('/') not in ['~', '.', '']
        U.f_remove(folder)
        print('Experiment folder "{}" removed.'.format(folder))


def error(group=None, window=None):
    _get_global_cluster().print_error(group, window)


def stdout(group=None, window=None, history=0):
    _get_global_cluster().print_stdout(group, window, history=history)