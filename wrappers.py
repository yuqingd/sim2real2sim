import atexit
import functools
import sys
import threading
import traceback
import time
import gym
import numpy as np
from PIL import Image
import cv2

from environments.reach import FetchReachEnv
from environments.push import FetchPushEnv
from environments.slide import FetchSlideEnv
from environments.kitchen.adept_envs.adept_envs.kitchen_multitask_v0 import KitchenTaskRelaxV1
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

from dm_control.mujoco import engine
from dm_control.rl.control import PhysicsError

class PegTask:
  def __init__(self, size=(64, 64), real_world=False, dr=None, use_state=False):
    from envs import Insert_XArm7Pos
    self._env = Insert_XArm7Pos()
    self._size = size
    self.real_world = real_world
    self.use_state = use_state
    self.dr = dr

    self.apply_dr()

  def apply_dr(self):
    pass

  @property
  def observation_space(self):
    spaces = {}
    if self.use_state:
      spaces['state'] = self._env.observation_space
    spaces['image'] = gym.spaces.Box(
      0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    state_obs, reward, done, info = self._env.step(action)
    obs = {}
    if self.use_state:
      obs['state'] = state_obs
    obs['image'] = self.render()
    info['discount'] = 1.0
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 1.0 if reward > 0 else 0.0
    return obs, reward, done, info

  def get_dr(self):
    return np.array([0])  # TODO: add this!

  def reset(self):
    self.apply_dr()
    state_obs = self._env.reset()
    obs = {}
    if self.use_state:
      obs['state'] = state_obs
    obs['image'] = self.render()
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 0.0
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    img = self._env.render(mode='rgb_array')
    return cv2.resize(img, self._size)


GOAL_DIM = 30
ARM_DIM = 13
XPOS_INDICES = {
    'arm': [4, 5, 6, 7, 8, 9, 10], #Arm,
    'end_effector': [10],
    'gripper': [11, 12, 13, 14, 15], #Gripper
    'knob_burner1': [22, 23],
    'knob_burner2': [24, 25],
    'knob_burner3': [26, 27],
    'knob_burner4': [28, 29],
    'light_switch': [32, 33],
    'slide': [38],
    'hinge': [41],
    'microwave': [44],
    'kettle': [47],
    'kettle_root': [48],

}

# For light task
BONUS_THRESH_LL = 0.3
BONUS_THRESH_HL = 0.3
#
#  0               world [ 0         0         0       ]
#  1     vive_controller [-0.44     -0.092     2.03    ]
#  2                     [ 0         0         1.8     ]
#  3       xarm_linkbase [ 0         0         1.8     ]
#  4               link1 [ 0         0         2.07    ]
#  5               link2 [ 0         0         2.07    ]
#  6               link3 [-1.23e-06  1.63e-05  2.36    ]
#  7               link4 [ 4.07e-05  0.0525    2.36    ]
#  8               link5 [ 0.000101  0.13      2.02    ]
#  9               link6 [ 0.000101  0.13      2.02    ]
# 10               link7 [ 0.00016   0.206     1.92    ] == end_effector
# 11  left_outer_knuckle [ 0.0352    0.206     1.86    ]
# 12         left_finger [ 0.0706    0.206     1.82    ]
# 13  left_inner_knuckle [ 0.0202    0.206     1.85    ]
# 14 right_outer_knuckle [-0.0348    0.206     1.86    ]
# 15        right_finger [-0.0703    0.206     1.82    ]
# 16 right_inner_knuckle [-0.0198    0.206     1.85    ]
# 17                desk [-0.1       0.75      0       ]
# 18           counters1 [-0.1       0.75      0       ]
# 19            counters [-0.1       0.75      0       ]
# 20                oven [-0.1       0.75      0       ]
# 21            ovenroot [ 0.015     0.458     0.983   ]
# 22              knob 1 [-0.133     0.678     2.23    ]
# 23            Burner 1 [ 0.221     0.339     1.59    ]
# 24              knob 2 [-0.256     0.678     2.23    ]
# 25            Burner 2 [-0.225     0.339     1.59    ]
# 26              knob 3 [-0.133     0.678     2.34    ]
# 27            Burner 3 [ 0.219     0.78      1.59    ]
# 28              knob 4 [-0.256     0.678     2.34    ]
# 29            Burner 4 [-0.222     0.78      1.59    ]
# 30            hoodroot [ 0         0.938     2.33    ]
# 31 lightswitchbaseroot [-0.4       0.691     2.28    ]
# 32     lightswitchroot [-0.4       0.691     2.28    ]
# 33    lightblock_hinge [-0.0044    0.638     2.19    ]
# 34            backwall [-0.1       0.75      0       ]
# 35            wallroot [-0.041     1.33      1.59    ]
# 36               wall2 [-1.41      0.204     1.59    ]
# 37        slidecabinet [ 0.3       1.05      2.6     ]
# 38               slide [ 0.3       1.05      2.6     ]
# 39           slidelink [ 0.075     0.73      2.6     ]
# 40        hingecabinet [-0.604     1.03      2.6     ]
# 41            hingecab [-0.604     1.03      2.6     ]
# 42       hingeleftdoor [-0.984     0.71      2.6     ]
# 43      hingerightdoor [-0.224     0.71      2.6     ]
# 44           microwave [-0.85      0.725     1.6     ]
# 45           microroot [-0.85      0.725     1.6     ]
# 46       microdoorroot [-1.13      0.455     1.79    ]
# 47              kettle [-0.269     0.35      1.63    ]
# 48          kettleroot [-0.269     0.35      1.63    ]

class Kitchen:
  def __init__(self, task='reach_kettle', size=(64, 64), real_world=False, dr=None, use_state=False, step_repeat=200,
               step_size=0.05, use_gripper=False, simple_randomization=False, dr_shape=None, outer_loop_version=0,
               control_version='mocap_ik', distance=2., azimuth=50, elevation=-40):
    if 'rope' in task:
      distance = 1.5
      azimuth = 40
      elevation = -40

    self._env = KitchenTaskRelaxV1(distance=distance, azimuth=azimuth, elevation=elevation, task_type=task)
    self.task = task
    self._size = size
    self.real_world = real_world
    self.use_state = use_state
    self.dr = dr
    self.step_repeat = step_repeat
    self.step_size = step_size
    if 'pick' in task:
      self.use_gripper = True
    else:
      self.use_gripper = False
    self.end_effector_name = 'end_effector'
    self.mocap_index = 3
    self.end_effector_index = 4
    if 'rope' in task:
      self.cylinder_index = 5
      self.box_with_hole_index = 6

    self.arm_njnts = 7
    self.simple_randomization = simple_randomization
    self.dr_shape = dr_shape
    self.outer_loop_version = outer_loop_version
    self.control_version = control_version

    self.apply_dr()

  def setup_task(self):
    init_xpos = self._env.sim.data.body_xpos

    if 'reach' in self.task:
      self.set_workspace_bounds('full_workspace')

      if self.task == 'reach_microwave':
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['microwave']])
      elif self.task == 'reach_slide':
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['slide']])
      elif self.task == 'reach_kettle':
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[-1] += 0.1  # goal in middle of kettle
      else:
        raise NotImplementedError

    elif 'push' in self.task:
      self.set_workspace_bounds('stove_area')

      if self.task == 'push_kettle_burner': #single goal test, push to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += 0.5
      else:
        self.goal = np.random.uniform(low=self.end_effector_bound_low, high=self.end_effector_bound_high) #randomly select goal location in workspace
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1] #set z pos to be same as kettle, since we only want to push in x,y

    elif 'slide' in self.task:
      self.set_workspace_bounds('front_stove_area')
      self.slide_d1 = None

      #decrease friction for sliding
      self._env.sim.model.geom_friction[220:225, :] = 0.002
      self._env.sim.model.geom_friction[97:104, :] = 0.002

      if self.task == 'slide_kettle_burner': #single goal test, slide to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += .6
      else:
        self.goal = np.random.uniform(low=[-1, 0, 0], high=[0, 1, 0]) #randomly select goal location in workspace OUTSIDE of end effector reach
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1] #set z pos to be same as kettle, since we only want to slide in x,y

    elif 'pick' in self.task:
      self.set_workspace_bounds('full_workspace')
      self.use_gripper = True
      self.orig_kettle_height = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1]
      self.pick_d1 = None

      if self.task == 'pick_kettle_burner': #single goal test, slide to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += 0.5
      else:
        self.goal = np.random.uniform(low=[-1, 0, 0], high=[0, 1, 0]) #randomly select goal location in workspace OUTSIDE of end effector reach
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1] #set z pos to be same as kettle, since we only want to slide in x,y

    elif 'rope' in self.task:
      self.set_workspace_bounds('no_restrictions')
      self.goal = self._env.sim.data.site_xpos[self.box_with_hole_index]


    else:
      raise NotImplementedError



  def get_reward(self):
    xpos = self._env.sim.data.body_xpos
    if 'reach' in self.task:
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])
      reward = -np.linalg.norm(end_effector - self.goal)
      done = np.abs(reward) < 0.25
    elif 'push' in self.task:
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])
        # two stage reward, first get to kettle, then kettle to goal
      kettle = np.squeeze(xpos[XPOS_INDICES['kettle']])
      kettlehandle = kettle.copy()
      #kettlehandle[-1] += 0.1  # goal in middle of kettle

      d1 = np.linalg.norm(end_effector - kettlehandle)
      d2 = np.linalg.norm(kettle - self.goal)
      done = np.abs(d2) < 0.25

      reward = -(d1 + d2)

    elif 'slide' in self.task:
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])
        # two stage reward, first get to kettle, then kettle to goal
      kettle = np.squeeze(xpos[XPOS_INDICES['kettle']])
      kettlehandle = kettle.copy()
      #kettlehandle[-1] += 0.  # goal in middle of kettle

      d1 = np.linalg.norm(end_effector - kettlehandle)
      if d1 < 0.1 and self.slide_d1 is None: #TODO: tune threshold for hitting kettle
        self.slide_d1 = d1

      d2 = np.linalg.norm(kettle - self.goal)
      done = np.abs(d2) < 0.25

      if self.slide_d1 is not None:
        reward = -(self.slide_d1 + d2)
      else:
        reward = -(d1 + d2)



    elif 'pick' in self.task:
      #three stage reward, first reach kettle, pick up kettle, then goal
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])

      kettle = np.squeeze(xpos[XPOS_INDICES['kettle']])
      kettlehandle = kettle.copy()
      kettlehandle[-1] += 0.15  # goal in middle of kettle
      kettle_height = kettle[-1]

      d1 = np.linalg.norm(end_effector - kettlehandle) #reach handle

      if d1 < 0.1 and self.pick_d1 is None: #TODO: tune this
        d1 = d1 - self._env.data.ctrl[self.arm_njnts] #TODO: scale gripper contribution to reward
        self.pick_d1 = d1

      d3 = np.linalg.norm(kettle[:2] - self.goal[:2]) #xy distance to goal

      if self.pick_d1 is not None:
        d2 = np.linalg.norm(self.orig_kettle_height - kettle_height) #distance kettle has been lifted
        if d2 > 0.02: #TODO: tune this
          #then we have lifted it
          d2 = -d2
        else:
          if d3 > 0.25: #TODO: tune this
          #then we haven't lifted it and it is far from goal, restart
            self.pick_d1 = None
            d2 = 0
      else:
        d2 = 0
      done = np.abs(d3) < 0.25

      reward = -(d1 + d2 + d3)

    elif 'rope' in self.task:
      cylinder_loc = self._env.sim.data.site_xpos[self.cylinder_index]
      reward = -np.linalg.norm(cylinder_loc - self.goal)
      done = np.abs(reward) < 0.05

    else:
      raise NotImplementedError

    return reward, done

  def get_sim(self):
    return self._env.sim

  def update_dr_param(self, param, param_name, eps=1e-3):
    if param_name in self.dr:
      mean, range = self.dr[param_name]
      range = max(range, eps)
      param[:] = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
      self.sim_params += [mean, range]

  def set_workspace_bounds(self, bounds):
    if bounds == 'no_restrictions':
      x_low = y_low = z_low = -float('inf')
      x_high = y_high = z_high = float('inf')
    elif bounds == 'full_workspace':
      x_low = -1.5  # Around the microwave
      x_high = 1.  # Around the sink
      y_low = -0.1  # Right in front of the robot's pedestal
      y_high = 2  # Past back burner
      z_low = 1.5  # Tabletop
      z_high = 5  # Cabinet height
    elif bounds == 'stove_area':
      x_low = -0.5  # Left edge of stove
      x_high = 0.5  # Right edge of stove
      y_low = -0.1  # Right in front of the robot's pedestal
      y_high = 1.0  # Back burner
      z_low = 1.5  # Tabletop
      z_high = 2.  # Around top of kettle
    elif bounds == 'front_stove_area':  # For use with sliding
      x_low = -0.5  # Left edge of stove
      x_high = 0.5  # Right edge of stove
      y_low = -0.1  # Right in front of the robot's pedestal
      y_high = 0.4  # Mid-front burner
      z_low = 1.5  # Tabletop
      z_high = 2.  # Around top of kettle
    else:
      raise NotImplementedError("No other bounds types")

    self.end_effector_bound_low = [x_low, y_low, z_low]
    self.end_effector_bound_high = [x_high, y_high, z_high]

  def apply_dr(self):
    self.sim_params = []
    if self.dr is None or self.real_world:
      if self.outer_loop_version == 1:
        self.sim_params = np.zeros(self.dr_shape)
      return  # TODO: start using XPOS_INDICES or equivalent for joints.

    if 'rope' in self.task:
      cylinder_viz = self._env.sim.model.geom_name2id('cylinder_viz')
      cylinder_body =self._env.sim.model.body_name2id('cylinder')
      self.update_dr_param(self._env.sim.model.dof_damping[0:1], 'joint1_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[1:2], 'joint2_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[2:3], 'joint3_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[3:4], 'joint4_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[4:5], 'joint5_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[5:6], 'joint6_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[6:7], 'joint7_damping')

      self.update_dr_param(self._env.sim.model.geom_rgba[cylinder_viz, 2], 'cylinder_b')
      self.update_dr_param(self._env.sim.model.geom_rgba[cylinder_viz, 1], 'cylinder_g')
      self.update_dr_param( self._env.sim.model.geom_rgba[cylinder_viz, 0], 'cylinder_r')
      self.update_dr_param(self._env.sim.model.body_mass[cylinder_body], 'cylinder_mass')

      self.update_dr_param(self._env.sim.model.light_diffuse[:3], 'lighting')

    else:
      self.update_dr_param(self._env.sim.model.dof_damping[0:1], 'joint1_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[1:2], 'joint2_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[2:3], 'joint3_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[3:4], 'joint4_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[4:5], 'joint5_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[5:6], 'joint6_damping')
      self.update_dr_param(self._env.sim.model.dof_damping[6:7], 'joint7_damping')
      self.update_dr_param(self._env.sim.model.geom_rgba[212:219, 2], 'kettle_b')
      if 'slide' not in self.task:
        self.update_dr_param(self._env.sim.model.geom_friction[220:225, 0], 'kettle_friction')
      self.update_dr_param(self._env.sim.model.geom_rgba[212:219, 1], 'kettle_g')
      self.update_dr_param(self._env.sim.model.body_mass[48:49], 'kettle_mass')
      self.update_dr_param(self._env.sim.model.geom_rgba[212:219, 0], 'kettle_r')
      self.update_dr_param(self._env.sim.model.body_mass[[22, 24, 26, 28]], 'knob_mass')
      self.update_dr_param(self._env.sim.model.light_diffuse[:3], 'lighting')
      self.update_dr_param(self._env.sim.model.geom_rgba[2:33:2, 2], 'robot_b')
      self.update_dr_param(self._env.sim.model.geom_friction[2:33, 0], 'robot_friction')
      self.update_dr_param(self._env.sim.model.geom_rgba[2:33:2, 1], 'robot_g')
      self.update_dr_param(self._env.sim.model.geom_rgba[2:33:2, 0], 'robot_r')
      self.update_dr_param(self._env.sim.model.geom_rgba[86:87, 2], 'stove_b')
      if 'slide' not in self.task:
        self.update_dr_param(self._env.sim.model.geom_friction[97:104, 0], 'stove_friction')
      self.update_dr_param(self._env.sim.model.geom_rgba[86:87, 1], 'stove_g')
      self.update_dr_param(self._env.sim.model.geom_rgba[86:87, 0], 'stove_r')


  def get_dr(self):
    if self.simple_randomization:
      return np.array([self._env.sim.model.body_mass[48]])
    if 'rope' in self.task:
      cylinder_viz = self._env.sim.model.geom_name2id('cylinder_viz')
      cylinder_body =self._env.sim.model.body_name2id('cylinder')
      arr = np.array([
        self._env.sim.model.dof_damping[0],
        self._env.sim.model.dof_damping[1],
        self._env.sim.model.dof_damping[2],
        self._env.sim.model.dof_damping[3],
        self._env.sim.model.dof_damping[4],
        self._env.sim.model.dof_damping[5],
        self._env.sim.model.dof_damping[6],
        self._env.sim.model.geom_rgba[cylinder_viz, 2],
        self._env.sim.model.geom_rgba[cylinder_viz, 1],
        self._env.sim.model.body_mass[cylinder_body],
        self._env.sim.model.geom_rgba[cylinder_viz, 0],
        self._env.sim.model.light_diffuse[0, 0]
      ])
    else:
      arr = np.array([
        self._env.sim.model.dof_damping[0],
        self._env.sim.model.dof_damping[1],
        self._env.sim.model.dof_damping[2],
        self._env.sim.model.dof_damping[3],
        self._env.sim.model.dof_damping[4],
        self._env.sim.model.dof_damping[5],
        self._env.sim.model.dof_damping[6],
        self._env.sim.model.geom_rgba[212, 2],
        self._env.sim.model.geom_friction[212, 0],
        self._env.sim.model.geom_rgba[212, 1],
        self._env.sim.model.body_mass[48],
        self._env.sim.model.geom_rgba[212, 0],
        self._env.sim.model.body_mass[22],
        self._env.sim.model.light_diffuse[0, 0],
        self._env.sim.model.geom_rgba[2, 2],
        self._env.sim.model.geom_friction[2, 0],
        self._env.sim.model.geom_rgba[2, 1],
        self._env.sim.model.geom_rgba[2, 0],
        self._env.sim.model.geom_rgba[86, 2],
        self._env.sim.model.geom_friction[86, 0],
        self._env.sim.model.geom_rgba[86, 1],
        self._env.sim.model.geom_rgba[86, 0],
      ])
    arr = arr.astype(np.float32)
    return arr


  @property
  def observation_space(self):
    spaces = {}

    if self.use_state:
      state_shape = 4 if self.use_gripper else 3  # 2 for fingers, 3 for end effector position
      state_shape = self.goal.shape + state_shape
      if 'kettle' in self.task:
        state_shape = 3 + state_shape #add in kettle xpos coodinates
      spaces['state'] = gym.spaces.Box(np.array([-float('inf')] * state_shape), np.array([-float('inf')] * state_shape))
    else:
      spaces['state'] = gym.spaces.Box(np.array([-float('inf')] * self.goal.shape[0]),
                                       np.array([float('inf')] * self.goal.shape[0]))
    spaces['image'] = gym.spaces.Box(
      0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    if self.control_version == 'dmc_ik':
      act_shape = 4 if self.use_gripper else 3  # 1 for fingers, 3 for end effector position
      return gym.spaces.Box(np.array([-100.0] * act_shape), np.array([100.0] * act_shape))
    elif self.control_version == 'mocap_ik':
      act_shape = 4 if self.use_gripper else 3  # 1 for fingers, 3 for end effector position
      return gym.spaces.Box(np.array([-1.0] * act_shape), np.array([1.0] * act_shape))
    else:
      return self._env.action_space


  def set_xyz_action(self, action):

    pos_delta = action * self.step_size
    new_mocap_pos = self._env.data.mocap_pos + pos_delta[None]#self._env.sim.data.site_xpos[self.end_effector_index].copy() + pos_delta[None]
    # new_mocap_pos = self._env.data.mocap_pos + pos_delta[None]

    new_mocap_pos[0, :] = np.clip(
      new_mocap_pos[0, :],
      self.end_effector_bound_low,
      self.end_effector_bound_high,
    )


    self._env.data.set_mocap_pos('mocap', new_mocap_pos)
    # self._env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))  # TODO: what's a quaternion?

  def set_gripper(self, action):
    #gripper either open or close
    cur_ac = self._env.data.qpos[self.arm_njnts] #current gripper position, either 0 or 0.85

    if action < 0:
      gripper_ac = 0 #open
    else:
      gripper_ac = 0.85 #close

    sequence = np.linspace(cur_ac, gripper_ac, num=50)
    for step in sequence:
      self._env.data.ctrl[self.arm_njnts] = step
      self._env.sim.step()
    #need to linearly space out control with multiple steps so simulator doesnt break


  def step(self, action):
    update = None
    if self.control_version == 'mocap_ik':
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.set_xyz_action(action[:3])

        if self.use_gripper:
          gripper_ac = action[-1]
          self.set_gripper(gripper_ac)


    elif self.control_version == 'dmc_ik':
      action = np.clip(action, self.action_space.low, self.action_space.high)
      xyz_pos = action[:3] * self.step_size + self._env.sim.data.site_xpos[self.end_effector_index]

      physics = self._env.sim
      # The joints which can be manipulated to move the end-effector to the desired spot.
      joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
      ikresult = qpos_from_site_pose(physics, self.end_effector_name, target_pos=xyz_pos, joint_names=joint_names, tol=1e-10, progress_thresh=10, max_steps=50)
      qpos = ikresult.qpos
      success = ikresult.success

      if success:
        action_dim = len(self._env.data.ctrl)
        qpos_low = self._env.model.jnt_range[:, 0]
        qpos_high = self._env.model.jnt_range[:, 1]
        update = np.clip(qpos[:action_dim], qpos_low[:action_dim], qpos_high[:action_dim])
        if self.use_gripper:
          # TODO: almost certainly not the right way to implement this
          gripper_pos = action[3:]
          update[-len(gripper_pos):] = gripper_pos
          raise NotImplementedError
        else:
          update[self.arm_njnts + 1:] = 0 #no gripper movement
        self._env.data.ctrl[:] = update

    elif self.control_version == 'position':
      update = np.clip(action, self.action_space.low, self.action_space.high)
    else:
      raise ValueError(self.control_version)

    if update is not None:
      self._env.data.ctrl[:] = update
    for _ in range(self.step_repeat):
        try:
            self._env.sim.step()
        except PhysicsError as e:
            success = False
            print("Physics error:", e)

    reward, done = self.get_reward()
    # if not success:
    #   reward = reward * 2
    #done = np.abs(reward) < 0.25   # TODO: tune threshold
    info = {}
    obs = {}
    if self.outer_loop_version == 1:
      obs['sim_params'] = self.sim_params
    obs['state'] = self.goal
    if self.use_state:
      obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.sim.data.body_xpos[XPOS_INDICES['kettle']]), np.squeeze(self._env.sim.data.site_xpos[self.end_effector_index])])
      if self.use_gripper:
        obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.data.qpos[self.arm_njnts])])
    obs['image'] = self.render()
    info['discount'] = 1.0
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 1.0 if done else 0.0
    return obs, reward, done, info


  def reset(self):
    self.apply_dr()
    self.setup_task()
    state_obs = self._env.reset()
    obs = {}
    obs['state'] = self.goal
    if self.outer_loop_version == 1:
      obs['sim_params'] = self.sim_params
    if self.use_state:
      obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.sim.data.body_xpos[XPOS_INDICES['kettle']]), np.squeeze(self._env.sim.data.site_xpos[self.end_effector_index])])
      if self.use_gripper:
        obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.data.qpos[self.arm_njnts])])  # TODO: compute gripper position, include it
    obs['image'] = self.render()
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 0.0
    return obs

  def render(self, size=None, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    if size is None:
        size = self._size
    h, w = size
    img = self._env.render(width=w, height=h, mode='rgb_array')
    return img

class MetaWorld:
  def __init__(self, name, size=(64, 64), real_world=False, dr=None, use_state=False):
    from metaworld.benchmarks import ML1
    self._env = ML1.get_train_tasks(name + "-v1")
    self._size = size
    self.real_world = real_world
    self.use_state = use_state
    self.dr = dr

    self.apply_dr()

  def apply_dr(self):
    pass

  @property
  def observation_space(self):
    spaces = {}
    if self.use_state:
      spaces['state'] = self._env.observation_space
    spaces['image'] = gym.spaces.Box(
      0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    state_obs, reward, done, info = self._env.step(action)
    time_out = self._env.active_env.curr_path_length == self._env.active_env.max_path_length
    done = done or time_out
    obs = {}
    if self.use_state:
      obs['state'] = state_obs[:3]  # Only include robot state
    obs['image'] = self.render()
    info['discount'] = 1.0
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 1.0 if info['success'] else 0.0
    return obs, reward, done, info

  def get_dr(self):
    return np.array([0])  # TODO: add this!

  def reset(self):
    self.apply_dr()
    state_obs = self._env.reset()
    obs = {}
    if self.use_state:
      obs['state'] = state_obs[:3]  # Only include robot state
    obs['image'] = self.render()
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 0.0
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    width, height = self._size
    return self._env.active_env.sim.render(mode='offscreen', width=width, height=height)

class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None, real_world=False, sparse_reward=True, dr=None, use_state=False,
                                     simple_randomization=False, dr_shape=None, outer_loop_type=0):

    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera
    self.real_world = real_world
    self.sparse_reward = sparse_reward
    self.use_state = use_state
    self.dr = dr
    self.simple_randomization = simple_randomization
    self.dr_shape = dr_shape
    self.outer_loop_version = outer_loop_type

    self.apply_dr()

  def apply_dr(self):
    self.sim_params = []
    if self.dr is None or self.real_world:
      if self.outer_loop_version == 1:
        self.sim_params = np.zeros(self.dr_shape)
      return
    if self.simple_randomization:
      mean, range = self.dr["ball_mass"]
      eps = 1e-3
      range = max(range, eps)
      self._env.physics.model.body_mass[2] = max(np.random.uniform(low=mean - range, high=mean + range), eps)
      self.sim_params.append(mean)
      self.sim_params.append(range)
    else:
      if "actuator_gain" in self.dr:
        mean, range = self.dr["actuator_gain"]
        eps = 1e-3
        range = max(range, eps)
        self._env.physics.model.actuator_gainprm[:, 0] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))
        self.sim_params.append(mean)
        self.sim_params.append(range)
      if "ball_mass" in self.dr:
        mean, range = self.dr["ball_mass"]
        eps = 1e-3
        range = max(range, eps)
        self._env.physics.model.body_mass[2] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))
        self.sim_params.append(mean)
        self.sim_params.append(range)
      # if "ball_size" in self.dr:
      #   mean, range = self.dr["ball_size"]
      #   eps = 1e-3
      #   self._env.physics.model.geom_rbound[-1] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))
      if "damping" in self.dr:
        mean, range = self.dr["damping"]
        eps = 1e-3
        range = max(range, eps)
        self._env.physics.model.dof_damping[:2] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))
        self.sim_params.append(mean)
        self.sim_params.append(range)
      if "friction" in self.dr:
        mean, range = self.dr["friction"]
        eps = 1e-6
        range = max(range, eps)
        # Only adjust sliding friction
        self._env.physics.model.geom_friction[:, 0] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))
        self.sim_params.append(mean)
        self.sim_params.append(range)
      # if "string_length" in self.dr:
      #   mean, range = self.dr["string_length"]
      #   eps = 1e-2
      #   self._env.physics.model.tendon_length0[0] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))
      # if "string_stiffness" in self.dr:
      #   mean, range = self.dr["string_stiffness"]
      #   eps = 0
      #   self._env.physics.model.tendon_stiffness[0] = np.random.uniform(low=max(mean-range, eps), high=max(mean+range, 2 * eps))


  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def get_dr(self):
    if self.simple_randomization:
      return np.array([self._env.physics.model.body_mass[2]])
    return np.array([
      self._env.physics.model.actuator_gainprm[0, 0],
      self._env.physics.model.body_mass[2],
      # self._env.physics.model.geom_rbound[-1],
      self._env.physics.model.dof_damping[0],
      self._env.physics.model.geom_friction[0, 0],
      # self._env.physics.model.tendon_length0[0],
      # self._env.physics.model.tendon_stiffness[0],
    ])

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    if self.use_state:
      obs['state'] = np.concatenate([obs['position'], obs['velocity']])  # TODO: these are specific to ball_in_cup. We should have a more general representation.  Also -- are these position and velocity of the ball or the cup?
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    if self.outer_loop_version == 1:
      obs['sim_params'] = self.sim_params
    info = {'discount': np.array(time_step.discount, np.float32)}
    obs['real_world'] = 1.0 if self.real_world else 0.0
    if self.outer_loop_version == 2:
      obs['dr_params'] = self.get_dr()
    if self.sparse_reward:
      obs['success'] = 1.0 if reward > 0 else 0.0
    return obs, reward, done, info

  def reset(self):
    self.apply_dr()
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    if self.use_state:
      obs['state'] = np.concatenate([obs['position'], obs['velocity']])
    obs['image'] = self.render()
    if self.outer_loop_version == 1:
      obs['sim_params'] = self.sim_params
    obs['real_world'] = 1.0 if self.real_world else 0.0
    if self.outer_loop_version == 2:
      obs['dr_params'] = self.get_dr()
    if self.sparse_reward:
      obs['success'] = 0.0
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Dummy:

  def __init__(self, name, size=(64, 64), camera=None, real_world=False, sparse_reward=True, dr=None, use_state=False):
    self.mass = 3.0
    self._size = size
    self.real_world = real_world
    self.left_half = np.random.uniform(low=-3., high=0.)
    self.right_half = np.random.uniform(low=-3., high=0.)
    self.dr = dr
    self.apply_dr()

  def apply_dr(self):
    if self.dr is None or self.real_world:
      return
    if "body_mass" in self.dr:
      mean, range = self.dr["body_mass"]
      eps = 1e-3
      self.mass = max(np.random.uniform(low=mean-range, high=mean+range), eps)


  @property
  def observation_space(self):
    spaces = {}
    # for key, value in self._env.observation_spec().items():
    #   spaces[key] = gym.spaces.Box(
    #       -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return gym.spaces.Box(np.array([-5, -5]), np.array([5, 5]), dtype=np.float32)

  def step(self, action):
    self.left_half += action[0] * .05
    self.right_half += self.mass * .01
    obs = {}
    obs['image'] = self.render()
    if np.abs(self.left_half - self.right_half) < .05:
      reward = 1.0
    else:
      reward = 0.0
    done = False
    info = {}
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['mass'] = self.mass
    obs['success'] = 1.0 if reward > 0 else 0.0
    return obs, reward, done, info

  def reset(self):
    self.apply_dr()
    self.left_half = np.random.uniform(low=-3., high=0.)
    self.right_half = np.random.uniform(low=-3., high=0.)
    obs = {}
    obs['image'] = self.render()
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['mass'] = self.mass
    obs['success'] = 0.0
    return obs

  def render(self, *args, **kwargs):
    rgb_array = np.zeros((64, 64, 3))
    rgb_array[:32] = self.left_half
    rgb_array[32:] = self.right_half
    return rgb_array


class GymControl:

  def __init__(self, name, size=(64, 64), camera=None, dr=None):
    if name == "FetchReach":
      FetchEnv = FetchReachEnv
    elif name == "FetchSlide":
      FetchEnv = FetchSlideEnv
    elif name == "FetchPush":
      FetchEnv = FetchPushEnv
    else:
      raise ValueError("Invalid env name " + name)
    generate_vision = True # TODO: pass in
    deterministic = False
    reward_type = "dense"
    distance_threshold = 0.05
    self._size = size
    if camera is None:
      camera = "external_camera_0" # TODO: need?
    self._camera = camera
    self.dr = dr

    if dr is not None:
      self._env = FetchEnv(use_vision=generate_vision, deterministic=deterministic, reward_type=reward_type,
                           distance_threshold=distance_threshold, real_world=False)
      self.apply_dr()
    else:
      self._env = FetchEnv(use_vision=generate_vision, deterministic=deterministic, reward_type=reward_type,
                           distance_threshold=distance_threshold, real_world=True)

  def apply_dr(self):
    if self.dr is None:
      return
    if "body_mass" in self.dr:
      mean, std = self.dr["body_mass"]
      eps = 1e-3
      self._env.sim.model.body_mass[32] = np.abs(np.random.normal(mean, std)) + eps

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_space.items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    # spec = self._env.action_space
    # return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return self._env.action_space

  def step(self, action):
    # time_step = self._env.step(action)
    # obs = dict(time_step.observation)
    obs, reward, done, info = self._env.step(action) # Done currently has None
    img = self.render()
    obs['image'] = img
    done = int(done)
    discount = 1 # TODO: discount?
    info = {'discount': np.array(discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    self.apply_dr()
    obs = self._env.reset()
    # time_step = self._env.reset()
    # obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    width, height = self._size
    return self._env.sim.render(width=width, height=height, camera_name=self._camera)[::-1]


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True):
    import gym
    version = 0 if sticky_actions else 4
    name = ''.join(word.title() for word in name.split('_'))
    with self.LOCK:
      self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
    self._action_repeat = action_repeat
    self._size = size
    self._grayscale = grayscale
    self._noops = noops
    self._life_done = life_done
    self._lives = None
    shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
    self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
    self._random = np.random.RandomState(seed=None)

  @property
  def observation_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.reset()
    noops = self._random.randint(1, self._noops + 1)
    for _ in range(noops):
      done = self._env.step(0)[2]
      if done:
        with self.LOCK:
          self._env.reset()
    self._lives = self._env.ale.lives()
    if self._grayscale:
      self._env.ale.getScreenGrayscale(self._buffers[0])
    else:
      self._env.ale.getScreenRGB2(self._buffers[0])
    self._buffers[1].fill(0)
    return self._get_obs()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      _, reward, done, info = self._env.step(action)
      total_reward += reward
      if self._life_done:
        lives = self._env.ale.lives()
        done = done or lives < self._lives
        self._lives = lives
      if done:
        break
      elif step >= self._action_repeat - 2:
        index = step - (self._action_repeat - 2)
        if self._grayscale:
          self._env.ale.getScreenGrayscale(self._buffers[index])
        else:
          self._env.ale.getScreenRGB2(self._buffers[index])
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self):
    if self._action_repeat > 1:
      np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
    image = np.array(Image.fromarray(self._buffers[0]).resize(
        self._size, Image.BILINEAR))
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    return {'image': image}


class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      print("VALUE", value)
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):


    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class ObsDict:

  def __init__(self, env, key='obs'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    if strategy != 'none':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=self._worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except ConnectionResetError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _worker(self, ctor, conn):
    try:
      env = ctor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError(f'Received message of unknown type {message}')
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error in environment process: {stacktrace}')
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()
