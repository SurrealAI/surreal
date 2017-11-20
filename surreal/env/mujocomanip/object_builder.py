from MujocoManip.model import *

def build_from_config(config):
    if config.type == 'xml_object':
        return MujocoXMLObject(config.xml)
    if config.type == 'default_box':
        return DefaultBoxObject()
    if config.type == 'default_ball':
        return DefaultBallObject()
    if config.type == 'box':
        return BoxObject(size=config.size, rgba=config.rgba)
    if config.type == 'randombox':
        return RandomBoxObject(size_max=config.size_max, size_min=size_min, seed=config.seed)
    raise ValueError('mujoco object type "{}" not recognized'.format(config.type))
