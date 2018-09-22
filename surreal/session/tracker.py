import itertools
from collections import deque
import surreal.utils as U
from tensorplex import TensorplexClient, LoggerplexClient
from threading import Lock
import time
import os


class PeriodicTracker(object):
    def __init__(self, period, init_value=0, init_endpoint=0):
        """
        first: if True, triggers at the first time
        """
        U.assert_type(period, int)
        assert period > 0
        U.assert_type(init_value, int)
        self.period = period
        self.value = init_value
        self._endpoint = init_endpoint

    def _update_endpoint(self):
        if self.value >= self._endpoint + self.period:
            end_incr = (self.value - self._endpoint) // self.period * self.period
            self._endpoint += end_incr
            return True
        else:
            return False

    def track_increment(self, incr=1):
        """
        Returns: True if we enter the next period
        """
        U.assert_type(incr, int)
        self.value += incr
        return self._update_endpoint()

    def track_absolute(self, value):
        """
        Returns: True if we enter the next period
        """
        U.assert_type(value, int)
        self.value = value
        return self._update_endpoint()


class RunningAverage(object):
    def __init__(self, gamma, init_value=None):
        """Keep a running estimate of a quantity. This is a bit like mean
        but more sensitive to recent changes.

        Parameters
        ----------
        gamma: float
            Must be between 0 and 1, where 0 is the most sensitive to recent
            changes.
        init_value: float or None
            Initial value of the estimate. If None, it will be set on the first update.
        """
        self._value = init_value
        self._gamma = gamma

    def update(self, new_val):
        """Update the estimate.

        Parameters
        ----------
        new_val: float
            new observated value of estimated quantity.
        """
        if self._value is None:
            self._value = new_val
        else:
            self._value = self._gamma * self._value + (1.0 - self._gamma) * new_val

    def __float__(self):
        """Get the current estimate"""
        return self._value


class TimeThrottledTensorplex(object):
    """
        A tensorplex client that aggregates output from multiple threads
        Thread safe

        Update criterion:
        1) A global step is provided when updating value
        2) Time from last update in more than min_update_interval
    """
    def __init__(self, tensorplex, min_update_interval):
        U.assert_type(tensorplex, TensorplexClient)
        self.tensorplex = tensorplex
        self.min_update_interval = min_update_interval
        self.history = U.AverageDictionary()
        self.lock = Lock()
        self.tracker = U.TimedTracker(self.min_update_interval)
        self.init_time = time.time()

    def add_scalars(self, tag_value_dict, global_step=None):
        with self.lock:
            self.history.add_scalars(tag_value_dict)
            if global_step is not None:
                if self.tracker.track_increment():
                    self.tensorplex.add_scalars(self.history.get_values(), global_step)


class PeriodicTensorplex(object):
    def __init__(self,
                 tensorplex,
                 period,
                 is_average=True,
                 keep_full_history=False):
        """
        Args:
            tensorplex: TensorplexClient object
            period: when you call `update()`, it will only send to Tensorplex
                at the specified period.
            is_average: if True, send the averaged value over the last `period`.
            keep_full_history: if False, only keep the last `period` of history.
        """
        if tensorplex is not None:  # None to turn off tensorplex
            U.assert_type(tensorplex, TensorplexClient)
        U.assert_type(period, int)
        assert period > 0
        self._tplex = tensorplex
        self._period = period
        self._is_average = is_average
        self._keep_full_history = keep_full_history
        self._tracker = PeriodicTracker(period)
        self._history = {}
        self._max_deque_size = None if keep_full_history else period

    def add_scalars(self, tag_value_dict, global_step=None):
        """

        Args:
            tag_value_dict:
            global_step: None to use the internal counter

        Returns:
            - None if period incomplete
            - dict of {tag: current_value (averaged)} at each period
        """
        for tag, value in tag_value_dict.items():
            if tag in self._history:
                self._history[tag].append(value)
            else:
                self._history[tag] = deque([value], maxlen=self._max_deque_size)
        if self._tracker.track_increment():
            current_values = {}
            for tag, history in self._history.items():
                if self._is_average:
                    history = itertools.islice(
                        history,
                        max(len(history) - self._period, 0),
                        len(history)
                    )  # simulate history[-period:] for deque
                    avg_value = U.mean(list(history))
                else:
                    avg_value = float(history[-1]) if len(history) > 0 else 0.0
                current_values[tag] = avg_value
            if self._tplex is not None:
                if global_step is None:
                    global_step = self._tracker.value
                self._tplex.add_scalars(current_values, global_step)
            return current_values

    def get_history(self):
        return {tag: list(history)
                for tag, history in self._history.items()}


def get_loggerplex_client(name, session_config):
    """
    Args:
        name: file name for the remote log file
        session_config: see session_config.loggerplex
    """
    C = session_config.loggerplex
    host = os.environ['SYMPH_LOGGERPLEX_HOST']
    port = os.environ['SYMPH_LOGGERPLEX_PORT']
    return LoggerplexClient(
        name,
        host=host,
        port=port,
        enable_local_logger=C.enable_local_logger,
        local_logger_stream='stdout',
        local_logger_level=C.local_logger_level,
        local_logger_time_format=C.local_logger_time_format,
    )


def get_tensorplex_client(client_id, session_config):
    host = os.environ['SYMPH_TENSORPLEX_HOST']
    port = os.environ['SYMPH_TENSORPLEX_PORT']
    return TensorplexClient(
        client_id,
        host=host,
        port=port
    )
