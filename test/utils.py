import surreal.utils as U
import pytest
import contextlib
import os, sys


@contextlib.contextmanager
def pytest_print_raises(exc):
    with pytest.raises(exc) as e_info:
        yield
    print('EXPECTED:', e_info)
