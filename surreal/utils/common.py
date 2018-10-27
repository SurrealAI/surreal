import sys
import os
import inspect
import collections
import functools
import argparse
import re
import pprint
from enum import Enum, EnumMeta
import time
from contextlib import contextmanager
from threading import Thread, Lock


def report_exitcode(code, name='process'):
    """
    Given exit code of a process, throw error / print result as desired
    """
    if code == 0:
        print('[Warning] {} exited with exit code 0'.format(name))
    elif code > 0:
        raise ValueError('[Warning] {} exited with exit code {}'.format(name, code))
    else:
        raise ValueError('[Warning] {} terminated by signal {}'.format(name, code))


def _get_qualified_type_name(type_):
    name = str(type_)
    r = re.compile("<class '(.*)'>")
    match = r.match(name)
    if match:
        return match.group(1)
    else:
        return name


def assert_type(x, expected_type, message=''):
    assert isinstance(x, expected_type), (
        (message + ': ' if message else '')
        + 'expected type `{}`, actual type `{}`'.format(
            _get_qualified_type_name(expected_type),
            _get_qualified_type_name(type(x))
        )
    )
    return True


def print_(*objs, h='', **kwargs):
    """
    Args:
      *objs: objects to be pretty-printed
      h: header string
      **kwargs: other kwargs to pass on to ``pprint()``
    """
    if h:
        print('=' * 20, h, '=' * 20)
    for obj in objs:
        print(pprint.pformat(obj, indent=4, **kwargs))
    if h:
        print('=' * (42 + len(h)))


class _GetItemEnumMeta(EnumMeta):
    """
    Hijack the __getitem__ method from metaclass, because subclass cannot
        override magic methods. More informative error message.
    """
    def __getitem__(self, option):
        enum_class = None
        for v in self.__members__.values():
            enum_class = v.__class__
            break
        assert enum_class is not None, \
            'must have at least one option in StringEnum'
        return get_enum(enum_class, option)


class StringEnum(Enum, metaclass=_GetItemEnumMeta):
    """
    https://docs.python.org/3.4/library/enum.html#duplicatefreeenum
    The created options will automatically have the same string value as name.

    Support [] subscript, i.e. MyFruit['orange'] -> MyFruit.orange
    """
    def __init__(self, *args, **kwargs):
        self._value_ = self.name


def create_string_enum(class_name, option_names):
    assert_type(option_names, str)
    assert_type(option_names, list)
    return StringEnum(class_name, option_names)


def get_enum(enum_class, option):
    """
    Args:
        enum_class:
        option: if the value doesn't belong to Enum, throw error.
            Can be either the str name or the actual enum value
    """
    assert issubclass(enum_class, StringEnum)
    if isinstance(option, enum_class):
        return option
    else:
        assert_type(option, str)
        option = option.lower()
        options = enum_class.__members__
        if option not in options:
            raise ValueError('"{}" is not a valid option for {}. '
                             'Available options are {}.'
             .format(option, enum_class.__name__, list(options)))
        return options[option]


def fformat(float_num, precision):
    """
    https://stackoverflow.com/a/44702621/3453033
    """
    assert isinstance(precision, int) and precision > 0
    return ('{{:.{}f}}'
            .format(precision)
            .format(float_num)
            .rstrip('0')
            .rstrip('.'))


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


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


def exclude_keys(exclude, d):
    """
    Remove the `exclude` keys from a kwargs dict.

    Args:
      exclude: list or set of keys to be excluded
      d: raw dict that might have irrelevant keys
    """
    assert is_sequence(exclude)
    return {k: v for k, v in d.items() if k not in set(exclude)}


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


class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print('elapsed', self.interval)


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


class AutoInitializeMeta(type):
    """
    Call the special method ._initialize() after __init__.
    Useful if some logic must be run after the object is constructed.
    For example, the following code doesn't work because `self.y` does not exist
    when super class calls self._initialize()

    class BaseClass():
        def __init__(self):
            self._initialize()

        def _initialize():
            self.x = self.get_x()

        def get_x(self):
            # abstract method that only subclass

    class SubClass(BaseClass):
        def __init__(self, y):
            super().__init__()
            self.y = y

        def get_x(self):
            return self.y * 3

    Fix:
    class BaseClass(metaclass=AutoInitializeMeta):
        def __init__(self):
            pass
            # self._initialize() is now automatically called after __init__

        def _initialize():
            print('INIT', self.x)

        def get_x(self):
            # abstract method that only subclass
            raise NotImplementedError
    """
    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        assert hasattr(obj, '_initialize'), \
            'AutoInitializeMeta requires that subclass implements _initialize()'
        obj._initialize()
        return obj


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

class MovingAverageRecorder():
    """
        Records moving average 
    """
    def __init__(self, decay=0.95):
        self.decay = decay
        self.cum_value = 0
        self.normalization = 0

    def add_value(self, value):
        self.cum_value *= self.decay
        self.cum_value += value

        self.normalization *= self.decay
        self.normalization += 1

        return self.cum_value / self.normalization

    def cur_value(self):
        """
            Returns current moving average, 0 if no records
        """
        if self.normalization == 0:
            return 0
        else:
            return self.cum_value / self.normalization

class ThreadSafeMovingAverageRecorder(MovingAverageRecorder):
    def __init__(self, decay=0.95):
        super().__init__()
        self.lock = Lock()

    def add_value(self, value):
        with self.lock:
            return super().add_value(value)

    def cur_value(self):
        with self.lock:
            return super().cur_value()

class TimeRecorder():
    """
        Records average of whatever context block it is recording
        Don't call time in two threads
    """
    def __init__(self, decay=0.9995, max_seconds=10):
        """
        Args:
            decay: Decay factor of smoothed moving average
                    Default is 0.9995, which is approximately moving average 
                    of 2000 samples
            max_seconds: round down all time differences larger than specified
                        Useful when the application just started and there are long waits 
                        that might throw off the average
        """
        self.moving_average = ThreadSafeMovingAverageRecorder(decay)
        self.max_seconds = max_seconds
        self.started = False

    @contextmanager
    def time(self):
        pre_time = time.time()
        yield None
        post_time = time.time()

        interval = min(self.max_seconds, post_time - pre_time)
        self.moving_average.add_value(interval)

    def start(self):
        if self.started:
            raise RuntimeError('Starting a started timer')
        self.pre_time = time.time()
        self.started = True

    def stop(self):
        if not self.started:
            raise RuntimeError('Stopping a timer that is not started')
        self.post_time = time.time()
        self.started = False
    
        interval = min(self.max_seconds, self.post_time - self.pre_time)
        self.moving_average.add_value(interval)

    def lap(self):
        if not self.started:
            raise RuntimeError('Stopping a timer that is not started')
        post_time = time.time()
    
        interval = min(self.max_seconds, post_time - self.pre_time)
        self.moving_average.add_value(interval)

        self.pre_time = post_time


    @property
    def avg(self):
        return self.moving_average.cur_value()


class PeriodicWakeUpWorker(Thread):
    """
    Args:
        @target: The function to be called periodically
        @interval: Time between two calls 
        @args: Args to feed to target()
        @kwargs: Key word Args to feed to target()
    """
    def __init__(self, target, interval=1, args=None, kwargs=None):
        Thread.__init__(self)
        self.target = target
        self.interval = interval
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self.args is None:
            self.args = []
        if self.kwargs is None:
            self.kwargs = {}
        while True:
            self.target(*self.args, **self.kwargs)
            time.sleep(self.interval)


class TimedTracker(object):
    def __init__(self, interval):
        self.init_time = time.time()
        self.last_time = self.init_time
        self.interval = interval

    def track_increment(self):
        cur_time = time.time()
        time_since_last = cur_time - self.last_time
        enough_time_passed = time_since_last >= self.interval
        if enough_time_passed:
            self.last_time = cur_time
        return enough_time_passed


class AverageValue(object):
    """
        Keeps track of average of things
        Always caches the latest value so no division by 0
    """
    def __init__(self, initial_value):
        self.last_val = initial_value
        self.sum = initial_value
        self.count = 1

    def add(self, value):
        self.last_val = value
        self.sum += value
        self.count += 1

    def avg(self, clear=True):
        """
            Get the average of the currently tracked value
        Args:
            @clear: if true (default), clears the cached sum/count
        """
        ans = self.sum / self.count
        if clear:
            self.sum = self.last_val
            self.count = 1
        return ans


class AverageDictionary(object):
    def __init__(self):
        self.data = {}

    def add_scalars(self, new_data):
        for key in new_data:
            if key in self.data:
                self.data[key].add(new_data[key])
            else:
                self.data[key] = AverageValue(new_data[key])

    def get_values(self, clear=True):
        response = {}
        for key in self.data:
            response[key] = self.data[key].avg(clear=clear)
        return response


def wait_for_popen(processes, verbose=True):
    """

    Wait for some processes, terminate all if one of them fails

    Args:
        processes: A list of Popen
        verbose (bool, default True): when True, print to console
            when a process exited with code 0

    Raises:
        RuntimeError: When one process fails
    """
    completed = [False for i in processes]
    while True:
        time.sleep(1)
        for i, process in enumerate(processes):
            if completed[i]:
                continue
            ret = process.poll()
            if ret is not None:
                if ret == 0:
                    completed[i] = True
                    if verbose:
                        print('Process {} exited with code 0'.format(i))
                else:
                    for process in processes:
                        if process is not None:
                            process.kill()
                    raise RuntimeError('Process {} exited with code {}'
                                       .format(i, ret))


def start_thread(func, daemon=True, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    t = Thread(
        target=func,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
    )
    t.start()
    return t
