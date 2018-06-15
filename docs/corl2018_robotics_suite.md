# CORL 2018 Code Samples: Surreal Robotics Suite

We develop the Robotics Suite in the MuJoCo physics engine, which simulates fast with multi-joint contact dynamics. It has been a favorable choice adopted by previous continuous control benchmarks. 

By the time of final release, we will provide OpenAI gym-style interfaces in Python 
with detailed API documentations, along with tutorials on how to import new robots 
and create new environments and new tasks. 

We plan to release this benchmark together with the Surreal distributed RL library to accelerate related research in robot learning.

We highlight four primary features in our suite: 

1. *Procedural Generation*: we provide a modularized API to procedurally generate combinations of robot models, arenas and parameterized 3D objects, which enables us to train policies with better robustness and generalization. 

2. *Controller Modes*: we support joint velocity control and position control to command the robots. 

3. *Multimodal Sensors*: we support heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception. 

4. *Teleoperation*: we support using 3D motion devices, such as VR controllers, to teleoperate the robots and collect human demonstrations.

All six tasks below extend from the `MujocoEnv` base class:

1. Block Lifting
2. Block Stacking
3. Bimanual Peg-in-hole
4. Bimanual Lifting
5. Bin Picking
6. Nut-and-peg Assembly

Thank you so much for your time! 

```python
class MujocoEnv():
    def __init__(self,
                 has_renderer=True,
                 render_collision_mesh=False,
                 render_visual_mesh=True,
                 control_freq=100,
                 horizon=500,
                 ignore_done=False,
                 use_camera_obs=False,
                 camera_name=None,
                 camera_height=256,
                 camera_width=256,
                 camera_depth=False,
                 demo_config=None,
                 **kwargs):
        """
        Initialize a Mujoco Environment
        
        Args:
            has_renderer: If true, render the simulation state in a viewer instead of headless mode.
            control_freq in Hz, how many control signals to receive in every second
            ignore_done: if True, never terminate the env
        """
        self.has_renderer = has_renderer
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.viewer = None

        # settings for camera observation
        self.use_camera_obs = use_camera_obs
        self.camera_name = camera_name
        if self.use_camera_obs and self.camera_name is None:
            raise ValueError('Must specify camera name when using camera obs')
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth
        self._reset_internal()

    ###########################################################
    # Publicly exposed methods to reset and step through the envs
    ###########################################################
    
    def initialize_time(self, control_freq):
        """
        Initialize the time constants used for simulation
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError('xml model defined non-positive time step')
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError('control frequency {} is invalid'.format(control_freq))
        self.control_timestep = 1 / control_freq

    def action_spec(self):
        raise NotImplementedError

    def reset(self):
        # if there is an active viewer window, destroy it
        self.close()
        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def step(self, action):
        reward = 0
        info = None
        if not self.done:
            self.t += 1
            self._pre_action(action)
            end_time = self.cur_time + self.control_timestep
            while self.cur_time < end_time:
                self.sim.step()
                self.cur_time += self.model_timestep
            reward, done, info = self._post_action(action)
            return self._get_observation(), reward, done, info
        else:
            raise ValueError('executing action in terminated episode')
            
    def reward(self, action):
        return 0

    def render(self, camera_id=0):
        self.viewer.render(camera_id=camera_id)

    def observation_spec(self):
        observation = self._get_observation()
        return observation

    def close(self):
        """
        Do any cleanup necessary here.
        """
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close() # change this to viewer.finish()?
            self.viewer = None
            
    ###########################################################
    # Methods customized for environment subclasses
    ###########################################################
    
    def reset_from_xml_string(self, xml_string):
        """
        Reloads the environment from an XML description of the environment.
        """

        # if there is an active viewer window, destroy it
        self.close()

        # load model from xml
        self.mjpy_model = load_model_from_xml(xml_string)

        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)
        if self.has_renderer:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
        else:
            render_context=MjRenderContextOffscreen(self.sim)
            render_context.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            render_context.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            self.sim.add_render_context(render_context)

        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.t = 0
        self.done = False

    def find_contacts(self, geoms_1, geoms_2):
        for contact in self.sim.data.contact[0:self.sim.data.ncon]:
            if (self.sim.model.geom_id2name(contact.geom1) in geoms_1 \
            and self.sim.model.geom_id2name(contact.geom2) in geoms_2) or \
            (self.sim.model.geom_id2name(contact.geom2) in geoms_1 \
            and self.sim.model.geom_id2name(contact.geom1) in geoms_2):
                yield contact

    ###########################################################
    # Internal methods
    ###########################################################
    
    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        self.model = None

    def _get_reference(self):
        """Set up necessary reference for objects"""
        pass

    def _reset_internal(self):
        # self.sim.set_state(self.sim_state_initial)
        self._load_model()
        self.mjpy_model = self.model.get_model(mode='mujoco_py')
        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)
        if self.has_renderer:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
        else:
            render_context=MjRenderContextOffscreen(self.sim)
            render_context.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            render_context.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            self.sim.add_render_context(render_context)

        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.t = 0
        self.done = False

    def _get_observation(self):
        """
            Returns an OrderedDict containing observations [(name_string, np.array), ...]
        """
        return OrderedDict()

    def _pre_action(self, action):
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        reward = self.reward(action)
        self.done = (self._check_terminated() or self.t >= self.horizon) and (not self.ignore_done)
        return reward, self.done, {}

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        return False
```
