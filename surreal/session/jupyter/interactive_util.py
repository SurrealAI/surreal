"""
Utils to create suites of interactive commands.
Use global variables to keep track of the instance.
"""
# convenient imports for initializing Jupyter
import os, sys, json, time, re, random, inspect, pickle
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


pp = U.print_
ploads = pickle.loads
pdumps = pickle.dumps


_SUITE = {}


def _check_globals(var_name, global_dict, var_class):
    assert var_name in global_dict, \
        'Variable "{}" does not exist in global scope.'.format(var_name)
    assert isinstance(global_dict[var_name], var_class), \
        ('Variable "{}" is not an instance of "{}" class.'
         .format(var_name, var_class))


def create_interactive_suite(suite_name, var_class):
    """
    Args:
        suite_name: internal suite key, default global var name, default
            set_global_var method name (e.g. `set_mysuite_var()`)
        var_class: global instance class
    """
    assert inspect.isclass(var_class)
    global _SUITE
    _SUITE[suite_name] = {
        'var_name': suite_name,
        'globals': None  # global dict
    }

    def set_global_var(global_dict, var_name):
        """
        User API:
        Specify the variable name of your global instance.
        The instance can be retrieved from globals()['var_name']

        Args:
            var_name: string of cluster variable name
            global_dict: pass in `globals()`
        """
        global _SUITE
        U.assert_type(var_name, str)
        U.assert_type(global_dict, dict)
        _check_globals(var_name, global_dict, var_class)
        GLOBAL = _SUITE[suite_name]
        GLOBAL['var_name'] = var_name
        GLOBAL['globals'] = global_dict

    def _get_global_var():
        """
        Internal API:
        Call _get_global_var() to retrieve the global instance in each of 
        your interactive methods.
        """
        global _SUITE
        GLOBAL = _SUITE[suite_name]
        if not GLOBAL['globals']:
            raise RuntimeError('Please call set_{}_var(globals(), ) first.'
                               .format(suite_name))
        var_name = GLOBAL['var_name']
        global_dict = GLOBAL['globals']
        _check_globals(var_name, global_dict, var_class)
        return global_dict[var_name]

    # give `set_global_var` a name specific to your suite, because it will be
    # called by the user.
    return set_global_var, _get_global_var
