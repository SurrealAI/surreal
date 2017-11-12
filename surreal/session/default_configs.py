from .config import extend_config


BASE_SESSION_CONFIG = {
    'redis': {
        'replay': {
            'name': 'replay',
            'host': '_str_',
            'port': '_int_',
        },
        'ps': {
            'name': 'ps',
            'host': '_str_',
            'port': '_int_',
        },
    },
    'log': {
        'file_name': None,
        'file_mode': 'w',
        'time_format': None,
        'print_level': 'INFO',
        'stream': 'out',
    },
}


LOCAL_SESSION_CONFIG = {
    'redis': {
        'replay': {
            'name': 'replay',
            'host': 'localhost',
            'port': 6379,
        },
        'ps': {
            'name': 'ps',
            'host': 'localhost',
            'port': 6379,
        },
    },
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)
