from surreal.session.config import extend_config

BASE_OBJECT_CONFIG = {
    'type': '_str_',
}

XML_OBJECT_CONFIG = {
    'type': 'xml_object',
    'xml': '_str_',
}
XML_OBJECT_CONFIG = extend_config(XML_OBJECT_CONFIG, BASE_OBJECT_CONFIG)

XML_BOX_CONFIG = {
    'type': 'default_box',
}
XML_BOX_CONFIG = extend_config(XML_BOX_CONFIG, BASE_OBJECT_CONFIG)


XML_BALL_CONFIG = {
    'type': 'default_ball',
}
XML_BALL_CONFIG = extend_config(XML_BALL_CONFIG, BASE_OBJECT_CONFIG)

DEFAULT_BOX_CONFIG = {
    'type': 'box',
    'size': [0.05, 0.05, 0.05],
    'rgba': [1, 0, 0, 1],
}
DEFAULT_BOX_CONFIG = extend_config(DEFAULT_BOX_CONFIG, BASE_OBJECT_CONFIG)

DEFAULT_RANDOM_BOX_CONFIG = {
    'type': 'random_box',
    'size_max': [0.07, 0.07, 0.07],
    'size_min': [0.03, 0.03, 0.03],
    'seed': '_int_',
}
DEFAULT_RANDOM_BOX_CONFIG = extend_config(DEFAULT_RANDOM_BOX_CONFIG, BASE_OBJECT_CONFIG)