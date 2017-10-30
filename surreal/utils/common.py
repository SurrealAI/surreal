import sys
import os
import threading
import inspect
import collections
import functools
import argparse
from easydict import EasyDict


class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        self._stop_event = threading.Event() # stoppable thread
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()


def assert_type(x, expected_type, message=''):
    assert isinstance(x, expected_type), (
        (message + ': ' if message else '')
        + 'expected type `{}`, actual type `{}`'.format(expected_type, type(x))
    )


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return (isinstance(obj, collections.Sequence)
            and not isinstance(obj, str))


def include_keys(include, d):
    """
    Pick out the `include` keys from a dict

    Args:
      include: list or set of keys to be included
      d: raw dict that might have irrelevant keys
    """
    assert is_sequence(include)
    return {k: v for k, v in d.items() if k in set(include)}


def iter_last(iterable):
    """
    For processing the last element differently
    Yields: (is_last=bool, element)
    """
    length = len(iterable)
    return ((i == length-1, x) for i, x in enumerate(iterable))


def _get_bound_args(func, *args, **kwargs):
    """
    https://docs.python.org/3/library/inspect.html#inspect.BoundArguments
    def f(a, b, c=5, d=6): pass
    get_bound_args(f, 3, 6, d=100) -> {'a':3, 'b':6, 'c':5, 'd':100}

    Returns:
        OrderedDict of bound arguments
    """
    arginfo = inspect.signature(func).bind(*args, **kwargs)
    arginfo.apply_defaults()
    return arginfo.arguments


class SaveInitArgsMeta(type):
    """
    Bounded arguments:
    https://docs.python.org/3/library/inspect.html#inspect.BoundArguments

    Store the captured constructor arguments to <instance>._init_args
    as OrderedDict. Can be retrieved by the property method <obj>.init_args
    """
    def __init__(cls, name, bases, attrs):
        # WARNING: must add class method AFTER super.__init__
        # adding attrs['new-method'] before __init__ has no effect!
        super().__init__(name, bases, attrs)
        @property
        def init_args(self):
            return self._init_args
        cls.init_args = init_args

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj._init_args = _get_bound_args(obj.__init__, *args, **kwargs)
        return obj


class SaveInitArgs(metaclass=SaveInitArgsMeta):
    """
    Either use metaclass hook:
        class MyObj(metaclass=SaveInitArgsMeta)
    or simply inherit
        class MyObj(SaveInitArgs)
    """
    pass


class noop_context:
    """
    Placeholder context manager that does nothing.
    We could have written simply as:

    @contextmanager
    def noop_context(*args, **kwargs):
        yield

    but the returned context manager cannot be called twice, i.e.
    my_noop = noop_context()
    with my_noop:
        do1()
    with my_noop: # trigger generator error
        do2()
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def meta_wrap(decor):
    """
    a decorator decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()

    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument.
      See stackoverflow: http://goo.gl/UEYbDB
    """
    single_callable = (lambda args, kwargs:
                       len(args) == 1 and len(kwargs) == 0 and callable(args[0]))
    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_wrap
def deprecated(func, msg='', action='warning'):
    """
    Function/class decorator: designate deprecation.

    Args:
      msg: string message.
      action: string mode
      - 'warning': (default) prints `msg` to stderr
      - 'noop': do nothing
      - 'raise': raise DeprecatedError(`msg`)
    """
    action = action.lower()
    if action not in ['warning', 'noop', 'raise']:
        raise ValueError('unknown action type {}'.format(action))
    if not msg:
        msg = 'This is a deprecated feature.'

    # only does the deprecation when being called
    @functools.wraps(func)
    def _deprecated(*args, **kwargs):
        if action == 'warning':
            print(msg, file=sys.stderr)
        elif action == 'raise':
            raise DeprecationWarning(msg)
        return func(*args, **kwargs)
    return _deprecated


def _trace_key(dict_trace, key):
    return 'key "{}" '.format('/'.join(dict_trace + [key]))


def _has_required(config):
    for key, val in config.items():
        if val == 'REQUIRED':
            return True
        elif isinstance(val, dict):
            if _has_required(val):
                return True
    return False


def _fill_default_config(config, default_config, dict_trace):
    for key, default_value in default_config.items():
        if key not in config:
            if default_value == 'REQUIRED':
                raise KeyError(_trace_key(dict_trace, key) + 'is a required config')
            elif isinstance(default_value, dict):
                if _has_required(default_value):
                    raise ValueError(_trace_key(dict_trace, key) + 'missing. '
                                                                   'Its sub-dict has a required config')
            config[key] = default_value
        else:
            value = config[key]
            if isinstance(value, dict) and not isinstance(default_value, dict):
                raise ValueError(_trace_key(dict_trace, key)
                                 + 'must be a single value instead of a sub-dict')
            if isinstance(default_value, dict):
                if not isinstance(value, dict):
                    raise ValueError(_trace_key(dict_trace, key)
                                     + 'must have a sub-dict instead of a single value')
                config[key] = _fill_default_config(value, default_value,
                                                   dict_trace + [key])
            if value == default_value == 'REQUIRED':
                raise ValueError(_trace_key(dict_trace, key) + ' is required.')
    return config


def fill_default_config(config, default_config):
    """
    Special: denote the value as 'REQUIRED' (all-caps) in default_config to enforce

    Returns:
        AttributeDict
        `config` filled by default values if certain keys are unspecified
    """
    return EasyDict(_fill_default_config(config, default_config, []))


class ArgParser(object):
    def __init__(self, **kwargs):
        """
        The following options are pre-configured
        --verbosity, or -vvv (number of v's indicate the level of verbosity)
        --debug: turn on debugging mode
        """
        kwargs['formatter_class'] = ArgParser._SingleMetavarFormatter
        self.parser = argparse.ArgumentParser(**kwargs)
        self.parser.add_argument('--verbose', '-v', action='count', default=-1,
                                 help='can repeat, e.g. -vvv for level 3 verbosity')
        self.parser.add_argument('--debug', action='store_true', default=False,
                                 help='Turn on debugging mode. ')

    def add(self, *args, **kwargs):
        default = kwargs.get('default')
        dtype = kwargs.get('type')
        if dtype is None:
            if default is None:
                dtype = str
            else:
                dtype = type(default)
        typename = dtype.__name__
        if 'metavar' not in kwargs:
            # metavar: display --foo <float=0.05> in help string
            if 'choices' in kwargs:
                choices = kwargs['choices']
                choices_str = '/'.join(['{}']*len(choices)).format(*choices)
                kwargs['metavar'] = '<{}: {}>'.format(typename, choices_str)
            elif 'nargs' in kwargs:
                # better formatting handled in _SingleMetavarFormatter
                kwargs['metavar'] = '{}'.format(typename)
            elif not kwargs.get('action'):
                # if 'store_true', then no metavar needed
                # list of actions: https://docs.python.org/3/library/argparse.html#action
                default_str = '={}'.format(default) if default else ''
                kwargs['metavar'] = '<{}{}>'.format(typename, default_str)
        self.parser.add_argument(*args, **kwargs)


    def add_boolean_flag(self, name, default=False, pair=True, help=None):
        """Add a boolean flag to argparse parser.

        Args:
            parser: argparse.Parser
                parser to add the flag to
            name: str
                --<name> will enable the flag, while --no-<name> will disable it
            default: bool or None
                default value of the flag
            pair:
                True to add both --myflag and --no-myflag
            help: str
                help string for the flag
        """
        self.parser.add_argument("--" + name,
                                 action="store_true", default=default, help=help)
        if pair:
            self.parser.add_argument("--no-" + name,
                                     action="store_false", dest=name)

    # aliases
    add_argument = add
    def parse(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)

    def __getattr__(self, attr):
        "delegate any other methods to the underlying parser"
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        else:
            return getattr(self.parser, attr)

    class _SingleMetavarFormatter(argparse.HelpFormatter):
        "Helper for better metavar display in ArgParser"
        def _format_action_invocation(self, action):
            if not action.option_strings:
                metavar, = self._metavar_formatter(action, action.dest)(1)
                return metavar
            else:
                parts = []
                # if the Optional doesn't take a value, format is `-s, --long`
                if action.nargs == 0:
                    parts.extend(action.option_strings)
                # if the Optional takes a value, format is:
                #    -s <METAVAR>, --long
                else:
                    default = action.dest.upper()
                    args_string = self._format_args(action, default)
                    ## THIS IS THE PART REPLACED
                    # for option_string in action.option_strings:
                    # parts.append('%s %s' % (option_string, args_string))
                    parts.extend(action.option_strings)
                    # treat nargs different
                    if action.nargs and action.default:
                        parts[-1] += ' default={}'.format(action.default)
                    parts[0] += ' ' + args_string
                return ', '.join(parts)
