ATARI_NAMES = [
    'air_raid',
    'alien',
    'amidar',
    'assault',
    'asterix',
    'asteroids',
    'atlantis',
    'bank_heist',
    'battle_zone',
    'beam_rider',
    'berzerk',
    'bowling',
    'boxing',
    'breakout',
    'carnival',
    'centipede',
    'chopper_command',
    'crazy_climber',
    'demon_attack',
    'double_dunk',
    'elevator_action',
    'enduro',
    'fishing_derby',
    'freeway',
    'frostbite',
    'gopher',
    'gravitar',
    'hero',
    'ice_hockey',
    'jamesbond',
    'journey_escape',
    'kangaroo',
    'krull',
    'kung_fu_master',
    'montezuma_revenge',
    'ms_pacman',
    'name_this_game',
    'phoenix',
    'pitfall',
    'pong',
    'pooyan',
    'private_eye',
    'qbert',
    'riverraid',
    'road_runner',
    'robotank',
    'seaquest',
    'skiing',
    'solaris',
    'space_invaders',
    'star_gunner',
    'tennis',
    'time_pilot',
    'tutankham',
    'up_n_down',
    'venture',
    'video_pinball',
    'wizard_of_wor',
    'yars_revenge',
    'zaxxon'
]


def _camelcase_names(names):
    return [''.join([s.capitalize() for s in name.split('_')])
            for name in names]


ATARI_NAMES_CAP = _camelcase_names(ATARI_NAMES)
_ATARI_CAP_MAP = {name.lower(): name for name in ATARI_NAMES_CAP}


def atari_name_cap(name):
    """
    Capitalize an Atari game name properly
    """
    name = name.lower()
    if name not in _ATARI_CAP_MAP:
        raise KeyError(name + ' is not a valid Atari game')
    return _ATARI_CAP_MAP[name]
