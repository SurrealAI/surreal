class PeriodicTracker(object):
    def __init__(self, period, init_value=0, init_endpoint=0):
        """
        first: if True, triggers at the first time
        """
        assert isinstance(period, int)
        assert isinstance(init_value, int)
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
        assert isinstance(incr, int)
        self.value += incr
        return self._update_endpoint()

    def track_absolute(self, value):
        """
        Returns: True if we enter the next period
        """
        assert isinstance(value, int)
        self.value = value
        return self._update_endpoint()


class RunningAvg(object):
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


