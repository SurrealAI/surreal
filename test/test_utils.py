import surreal.utils as U
import pytest
from test.utils import *


def test_enum():
    class MyMode(U.StringEnum):
        apple = ()
        orange = ()

    with pytest_print_raises(ValueError):
        m1 = MyMode['banana']

    assert MyMode[MyMode.apple] == MyMode.apple
    assert MyMode['orange'] == MyMode.orange
