from surreal.session import *
import copy
import pytest
import contextlib


@contextlib.contextmanager
def pytest_print_raises(exc):
    with pytest.raises(exc) as e_info:
        yield
    print('EXPECTED:', e_info)


@pytest.fixture
def C():
    C = {
        'redis': {
            'replay': {
                'host': 'localhost',
                'port': 6379,
            },
            'ps': {
                'host': '_dict_',
                'port': '_list_',
                'single': '_singleton_',
            },
        },
        'log': {
            'files': ['f1.txt', 'f2.txt', 'f3.txt'],
            'outputs': [
                {'stdout1': 1, 'stdout2': 2},
                {'stderr1': 10, 'stderr2': 20}
            ],
        },
    }
    return Config(C)


@pytest.fixture
def C_num():
    C = {
        'redis': {
            'replay': {
                'host': '_str_',
                'port': '_num_',
                'catchall': '_object_'
            },
            'ps': {
                'fport': '_float_',
                'iport': '_int_',
                'flag': '_bool_'
            },
        },
    }
    return Config(C)

@pytest.fixture
def C_extended():
    return {'log': {'files': ['f1.txt', 'f2.txt', 'f3.txt'],
             'outputs': [{'stdout2': 2, 'stdout1': 1},
                         {'stderr1': 10, 'stderr2': 20}]},
     'redis': {'replay': {'host': 'localhost', 'port': 6379},
               'ps': {'host': {'s': 2}, 'port': [1, 2], 'single': 'one-value'}}}


def test_access_key(C):
    print(
        C.redis.replay.host,
        C.log.files[1],
        C.log.outputs[1].stderr2
    )

# @pytest.mark.xfail(raises=ConfigError, strict=True)
def test_bad_key(C):
    with pytest.raises(ConfigError):
        print(C.redis.ps.badkey)


def test_default_dict(C, C_extended):
    C = extend_config({
        'redis': {
            'ps': {'host': {'s': 2}, 'port': [1, 2], 'single': 'one-value'}
        }
    }, C)
    assert C == C_extended


def test_default_dict_extend(C, C_extended):
    C2 = Config({
        'redis': {
            'ps': {'host': {'s': 2}, 'port': [1, 2], 'single': 'one-value'}
        }
    })
    C2.extend(C)
    assert C2 == C2.to_dict() == C_extended


def test_bad_req_dict(C):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'ps': {'host': 3, 'port': [1, 2]}
            }
        }, C)


def test_bad_req_list(C):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'ps': {'host': {'s': 2}, 'port': {'t': 'damn'}}
            }
        }, C)


def test_bad_req_single(C):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'ps': {'host': {'s':2}, 'port': [1, 2], 'single': {}},
            }
        }, C)


def test_bad_single(C):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'ps': {'host': {'s':2}, 'port': [1, 2], 'single': 'one-value'},
                'replay': 'wrong single value'
            }
        }, C)


def test_bad_dict(C):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'ps': {'host': {'s':2}, 'port': [1, 2], 'single': 'one-value'},
                'replay': {'host': {}}
            }
        }, C)


def test_missing_req(C):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'ps': {'port': [1, 2], 'single': 'one-value'},
            }
        }, C)

def recursive_print(D):
    for v in D.values():
        print(type(v))
        if isinstance(v, dict):
            recursive_print(v)


def test_json(C):
    Config(C).dump_file('test.json')
    assert Config.load_file('test.json') == C


def test_yaml(C):
    Config(C).dump_file('test.yaml')
    assert Config.load_file('test.yaml') == C


def test_num_error(C_num):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'replay': {
                    'host': 3,  # error
                    'port': 123,
                    'catchall': None
                },
                'ps': {
                    'fport': 1.23,
                    'iport': 10,
                    'flag': False
                },
            },
        }, C_num)


def test_int_error(C_num):
    with pytest_print_raises(ConfigError):
        extend_config({
            'redis': {
                'replay': {
                    'host': 'localhost',
                    'port': 123.5,
                    'catchall': None
                },
                'ps': {
                    'fport': 1.23,
                    'iport': 10.78,  # error
                    'flag': False
                },
            },
        }, C_num)


def test_all_types(C_num):
    _C_correct = {
        'redis': {
            'replay': {
                'host': 'localhost',
                'port': 13.23,
                'catchall': None
            },
            'ps': {
                'fport': 1.2e4,
                'iport': 10,
                'flag': False
            },
        },
    }
    C_correct = copy.deepcopy(_C_correct)
    assert extend_config(_C_correct, C_num) == C_correct


def test_enum_options():
    C_default = {
        'redis': {
            'replay': {
                'type': '_enum[uniform, priority, fifo]_'
            }
        }
    }
    C = {
        'redis': {
            'replay': {
                'type': 'fifo'
            }
        }
    }
    C_correct = copy.deepcopy(C)
    assert extend_config(C, C_default) == C_correct

