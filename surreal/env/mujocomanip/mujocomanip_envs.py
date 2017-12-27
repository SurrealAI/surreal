from MujocoManip import *
from surreal.env.base import Env
from surreal.env.mujocomanip.object_builder import build_from_config

class SurrealSawyerPushEnv(SawyerPushEnv, Env):
    def __init__(self, config):
        mujoco_object = build_from_config(config.mujoco_object_spec)
        super().__init__(mujoco_object=mujoco_object, **config)

    def _reset(self):
        return super()._reset(), {}

class SurrealSawyerStackEnv(SawyerStackEnv, Env):
    def __init__(self, config):
        mujoco_objects = [build_from_config(x) for x in config.mujoco_objects_spec]
        super().__init__(mujoco_objects=mujoco_objects,**config)

    def _reset(self):
        return super()._reset(), {}

class SurrealSawyerGraspEnv(SawyerGraspEnv, Env):
    def __init__(self, config):
        mujoco_object = build_from_config(config.mujoco_object_spec)
        super().__init__(mujoco_object=mujoco_object, **config)

    def _reset(self):
        return super()._reset(), {}
# TODO: add error checks for stuff like action spec