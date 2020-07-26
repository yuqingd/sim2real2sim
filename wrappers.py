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
    'gripper': [11, 12, 13, 14, 15, 16], #Gripper
    'knob_burner1': [22, 23],
    'knob_burner2': [24, 25],
    'knob_burner3': [26, 27],
    'knob_burner4': [28, 29],
    'light_switch': [31, 32],
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
  def __init__(self, task='reach_kettle', size=(64, 64), real_world=False, dr=None, mean_only=False,
               early_termination=False, use_state=False, step_repeat=200, dr_list=[],
               step_size=0.05, simple_randomization=False, dr_shape=None, outer_loop_version=0,
               control_version='mocap_ik', distance=2., azimuth=50, elevation=-40):
    if 'rope' in task:
      distance = 1.5
      azimuth = 20
      elevation = -20
    if 'cabinet' in task:
      distance = 2.5
      azimuth = 120
      elevation = -40
    if 'open_microwave' in task:
      distance = 1.5
      azimuth = 140
      elevation = -30

    self._env = KitchenTaskRelaxV1(distance=distance, azimuth=azimuth, elevation=elevation, task_type=task)
    self.task = task
    self._size = size
    self.early_termination = early_termination
    self.mean_only = mean_only
    self.real_world = real_world
    self.use_state = use_state
    self.dr = dr
    self.step_repeat = step_repeat
    self.step_size = step_size
    self.dr_list = dr_list
    if 'pick' in task or  'microwave' in task:
      self.use_gripper = True
    else:
      self.use_gripper = False
    self.end_effector_name = 'end_effector'
    self.mocap_index = 3
    self.end_effector_index = self._env.sim.model._site_name2id['end_effector']
    if 'rope' in task:
      self.cylinder_index = 5
      self.box_with_hole_index = 6

    self.arm_njnts = 7
    self.simple_randomization = simple_randomization
    self.dr_shape = dr_shape
    self.outer_loop_version = outer_loop_version
    self.control_version = control_version

    self.has_kettle = False if 'open_microwave' in task else True

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
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal += np.random.normal(loc=0, scale=0.3) #randomly select goal location in workspace
        self.goal[1] += 0.4 # move forward in y
        self.goal = np.clip(self.goal, self.end_effector_bound_low, self.end_effector_bound_high)
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1] #set z pos to be same as kettle, since we only want to push in x,y

    elif 'slide' in self.task:
      self.set_workspace_bounds('front_stove_area')
      self.slide_d1 = None

      #decrease friction for sliding
      self._env.sim.model.geom_friction[220:225, :] = 0.002
      self._env.sim.model.geom_friction[97:104, :] = 0.002

      if self.task == 'slide_kettle_burner': #single goal test, slide to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += .5
      else:
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal += np.random.normal(loc=0, scale=0.3) #randomly select goal location in workspace
        self.goal[1] += 0.5  # move forward in y
        self.goal = np.clip(self.goal, [-.5, 0.45, 0], [.5, 1, 0])
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1]  # set z pos to be same as kettle, since we only want to push in x,y

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

    elif 'open_microwave' in self.task:
      self.set_workspace_bounds('full_workspace')
      self.goal = np.squeeze(init_xpos[XPOS_INDICES['microwave']]).copy()
      self.goal += 0.5
    elif 'open_cabinet' in self.task:
      self.set_workspace_bounds('full_workspace')
      img_orig = self.render(size=(512, 512)).copy()

      goal = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']].copy()
      self.goal = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']].copy()
      self.goal[0] = 0.18
      self.step(np.array([0, 0, 0]))
      end_effector = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']].copy()
      ratio_to_goal = 0.6
      partway = ratio_to_goal * goal + (1 - ratio_to_goal) * end_effector
      for i in range(60):
        diff = partway - self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']].copy()
        diff = diff / self.step_size
        self.step(diff)

    else:
      raise NotImplementedError(self.task)



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
      if d1 < 0.35 and self.slide_d1 is None: #TODO: tune threshold for hitting kettle
        self.slide_d1 = d1


      d2 = np.linalg.norm(kettle - self.goal)
      done = np.abs(d2) < 0.2

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
      done = np.abs(reward) < 0.1

    elif 'open_microwave' in self.task:
      end_effector = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']]
      microwave_pos = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door']]
      microwave_pos_top = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door_top']]
      microwave_pos_bottom = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door_bottom']]

      # Always give a reward for having the end-effector near the shelf
      # since multiple z-positions are valid we'll compute each dimension separately
      x_dist = abs(end_effector[0] - microwave_pos[0])
      y_dist = abs(end_effector[1] - microwave_pos[1])
      if end_effector[2] > microwave_pos_top[2]:
        z_dist = end_effector[2] - microwave_pos_top[2]
      elif end_effector[2] < microwave_pos_bottom[2]:
        z_dist = microwave_pos_bottom[2] - end_effector[2]
      else:
        z_dist = 0
      dist_to_handle = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
      reach_rew = -dist_to_handle

      # Also have a reward for moving the cabinet
      dist_to_goal = np.abs(microwave_pos[1] - 0.28)
      move_rew = -dist_to_goal
      reward = reach_rew + move_rew

      done = dist_to_goal < 0.05

    elif 'open_cabinet' in self.task:
      end_effector = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']]
      cabinet_pos = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']]
      cabinet_pos_top = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door_top']]
      cabinet_pos_bottom = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door_bottom']]


      # Always give a reward for having the end-effector near the shelf
      # since multiple z-positions are valid we'll compute each dimension separately
      x_dist = abs(end_effector[0] - cabinet_pos[0])
      y_dist = abs(end_effector[1] - cabinet_pos[1])
      if end_effector[2] > cabinet_pos_top[2]:
        z_dist = end_effector[2] - cabinet_pos_top[2]
      elif end_effector[2] < cabinet_pos_bottom[2]:
        z_dist = cabinet_pos_bottom[2] - end_effector[2]
      else:
        z_dist = 0
      dist_to_handle = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
      reach_rew = -dist_to_handle

      # Also have a reward for moving the cabinet
      dist_to_goal = np.abs(cabinet_pos[0] - 0.18)
      move_rew = -dist_to_goal
      reward = reach_rew + move_rew

      done = dist_to_goal < 0.05

    else:
      raise NotImplementedError

    return reward, done

  def get_sim(self):
    return self._env.sim

  def update_dr_param(self, param, param_name, eps=1e-3, indices=None):
    if param_name in self.dr:
      if self.mean_only:
        mean = self.dr[param_name]
        range = max(0.1 * mean, eps) #TODO: tune this?
      else:
        mean, range = self.dr[param_name]
        range = max(range, eps)
      new_value = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
      if indices is None:
        param[:] = new_value
      else:
        try:
          for i in indices:
            param[i:i+1] = new_value
        except:
          param[indices:indices+1] = new_value

      if self.mean_only:
        self.sim_params += [mean]
      else:
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
      geom_dict = self._env.sim.model._geom_name2id
      xarm_viz_indices = 2
      model = self._env.sim.model
      # cylinder
      cylinder_viz = self._env.sim.model.geom_name2id('cylinder_viz')
      cylinder_body = self._env.sim.model.body_name2id('cylinder')

      # box
      box_viz_1 = self._env.sim.model.geom_name2id('box_viz_1')
      box_viz_2 = self._env.sim.model.geom_name2id('box_viz_2')
      box_viz_3 = self._env.sim.model.geom_name2id('box_viz_3')
      box_viz_4 = self._env.sim.model.geom_name2id('box_viz_4')
      box_viz_5 = self._env.sim.model.geom_name2id('box_viz_5')
      box_viz_6 = self._env.sim.model.geom_name2id('box_viz_6')
      box_viz_7 = self._env.sim.model.geom_name2id('box_viz_7')
      box_viz_8 = self._env.sim.model.geom_name2id('box_viz_8')

      dr_update_dict = {
        'joint1_damping': (model.dof_damping[0:1], None),
        'joint2_damping': (model.dof_damping[1:2], None),
        'joint3_damping': (model.dof_damping[2:3], None),
        'joint4_damping': (model.dof_damping[3:4], None),
        'joint5_damping': (model.dof_damping[4:5], None),
        'joint6_damping': (model.dof_damping[5:6], None),
        'joint7_damping': (model.dof_damping[6:7], None),
        'robot_r': (model.geom_rgba[:, 0], xarm_viz_indices),
        'robot_g': (model.geom_rgba[:, 1], xarm_viz_indices),
        'robot_b': (model.geom_rgba[:, 2], xarm_viz_indices),
        'cylinder_r': (model.geom_rgba[:, 0], cylinder_viz),
        'cylinder_g': (model.geom_rgba[:, 1], cylinder_viz),
        'cylinder_b': (model.geom_rgba[:, 2], cylinder_viz),
        'cylinder_mass': (model.body_mass[cylinder_body:cylinder_body+1], None),

        'box1_r': (model.geom_rgba[:, 0], box_viz_1),
        'box1_g': (model.geom_rgba[:, 1], box_viz_1),
        'box1_b': (model.geom_rgba[:, 2], box_viz_1),
        'box2_r': (model.geom_rgba[:, 0], box_viz_2),
        'box2_g': (model.geom_rgba[:, 1], box_viz_2),
        'box2_b': (model.geom_rgba[:, 2], box_viz_2),
        'box3_r': (model.geom_rgba[:, 0], box_viz_3),
        'box3_g': (model.geom_rgba[:, 1], box_viz_3),
        'box3_b': (model.geom_rgba[:, 2], box_viz_3),
        'box4_r': (model.geom_rgba[:, 0], box_viz_4),
        'box4_g': (model.geom_rgba[:, 1], box_viz_4),
        'box4_b': (model.geom_rgba[:, 2], box_viz_4),
        'box5_r': (model.geom_rgba[:, 0], box_viz_5),
        'box5_g': (model.geom_rgba[:, 1], box_viz_5),
        'box5_b': (model.geom_rgba[:, 2], box_viz_5),
        'box6_r': (model.geom_rgba[:, 0], box_viz_6),
        'box6_g': (model.geom_rgba[:, 1], box_viz_6),
        'box6_b': (model.geom_rgba[:, 2], box_viz_6),
        'box7_r': (model.geom_rgba[:, 0], box_viz_7),
        'box7_g': (model.geom_rgba[:, 1], box_viz_7),
        'box7_b': (model.geom_rgba[:, 2], box_viz_7),
        'box8_r': (model.geom_rgba[:, 0], box_viz_8),
        'box8_g': (model.geom_rgba[:, 1], box_viz_8),
        'box8_b': (model.geom_rgba[:, 2], box_viz_8),

        'rope_damping': (model.tendon_damping, None),
        'rope_friction': (model.tendon_frictionloss, None),
        'rope_stiffness': (model.tendon_stiffness, None),

        'lighting': (model.light_diffuse[:3], None),
      }
      for dr_param in self.dr_list:
        arr, indices = dr_update_dict[dr_param]
        self.update_dr_param(arr, dr_param, indices=indices)

    else:
      geom_dict = self._env.sim.model._geom_name2id
      stove_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "stove_collision" in name]
      stove_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "stove_viz" in name]
      xarm_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "xarm_viz" in name]
      xarm_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "xarm_collision" in name or "end_effector" in name]
      data = self._env.sim.data
      model = self._env.sim.model

      # Cabinet
      cabinet_index = self._env.sim.model.body_name2id('slidelink')
      cabinet_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_viz" in name]
      cabinet_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_collision" in name]

      # Microwave
      microwave_index = self._env.sim.model.body_name2id('microdoorroot')
      microwave_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_viz" in name]
      microwave_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_collision" in name]

      dr_update_dict = {
        'joint1_damping': (model.dof_damping[0:1], None),
        'joint2_damping': (model.dof_damping[1:2], None),
        'joint3_damping': (model.dof_damping[2:3], None),
        'joint4_damping': (model.dof_damping[3:4], None),
        'joint5_damping': (model.dof_damping[4:5], None),
        'joint6_damping': (model.dof_damping[5:6], None),
        'joint7_damping': (model.dof_damping[6:7], None),
        'cabinet_r': (model.geom_rgba[:, 0], cabinet_viz_indices),
        'cabinet_g': (model.geom_rgba[:, 1], cabinet_viz_indices),
        'cabinet_b': (model.geom_rgba[:, 2], cabinet_viz_indices),
        'cabinet_friction': (model.geom_friction[:, 0], cabinet_collision_indices),
        'cabinet_mass': (model.body_mass[cabinet_index: cabinet_index + 1], None),

        'knob_mass': (model.body_mass, [22, 24, 26, 28]),
        'lighting': (model.light_diffuse[:3], None),

        'microwave_r': (model.geom_rgba[:, 0], microwave_viz_indices),
        'microwave_g': (model.geom_rgba[:, 1], microwave_viz_indices),
        'microwave_b': (model.geom_rgba[:, 2], microwave_viz_indices),
        'microwave_friction': (model.geom_friction[:, 0], microwave_collision_indices),
        'microwave_mass': (model.body_mass[microwave_index: microwave_index + 1], None),
        'robot_r': (model.geom_rgba[:, 0], xarm_viz_indices),
        'robot_g': (model.geom_rgba[:, 1], xarm_viz_indices),
        'robot_b': (model.geom_rgba[:, 2], xarm_viz_indices),
        'stove_r': (model.geom_rgba[:, 0], stove_viz_indices),
        'stove_g': (model.geom_rgba[:, 1], stove_viz_indices),
        'stove_b': (model.geom_rgba[:, 2], stove_viz_indices),
        'stove_friction': (model.geom_friction[:, 0], stove_collision_indices),

      }

      # Kettle
      if self.has_kettle:
        kettle_index = self._env.sim.model.body_name2id('kettleroot')
        kettle_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "kettle_viz" in name]

        dr_update_dict_k = {
        'kettle_r': (model.geom_rgba[:, 0], kettle_viz_indices),
        'kettle_g': (model.geom_rgba[:, 1], kettle_viz_indices),
        'kettle_b': (model.geom_rgba[:, 2], kettle_viz_indices),
        'kettle_friction': (model.geom_friction[:, 0], kettle_viz_indices),
        'kettle_mass': (model.body_mass[kettle_index: kettle_index + 1], None),

        }
        dr_update_dict.update(dr_update_dict_k)


      # Actually Update
      for dr_param in self.dr_list:
        arr, indices = dr_update_dict[dr_param]
        self.update_dr_param(arr, dr_param, indices=indices)


  def get_dr(self):
    if self.simple_randomization:
      if 'rope' in self.task:
        cylinder_body = self._env.sim.model.body_name2id('cylinder')
        return np.array([self._env.sim.model.body_mass[cylinder_body]])
      elif 'open_microwave' in self.task:
        microwave_index = self._env.sim.model.body_name2id('microdoorroot')
        return np.array([self._env.sim.model.body_mass[microwave_index]])
      elif 'open_cabinet' in self.task:
        cabinet_index = self._env.sim.model.body_name2id('slidelink')
        return np.array([self._env.sim.model.body_mass[cabinet_index]])
      else:
        kettle_index = self._env.sim.model.body_name2id('kettleroot')
        return np.array([self._env.sim.model.body_mass[kettle_index]])
    if 'rope' in self.task:
      geom_dict = self._env.sim.model._geom_name2id
      cylinder_viz = self._env.sim.model.geom_name2id('cylinder_viz')
      cylinder_body =self._env.sim.model.body_name2id('cylinder')
      box_viz_1 = self._env.sim.model.geom_name2id('box_viz_1')
      box_viz_2 = self._env.sim.model.geom_name2id('box_viz_2')
      box_viz_3 = self._env.sim.model.geom_name2id('box_viz_3')
      box_viz_4 = self._env.sim.model.geom_name2id('box_viz_4')
      box_viz_5 = self._env.sim.model.geom_name2id('box_viz_5')
      box_viz_6 = self._env.sim.model.geom_name2id('box_viz_6')
      box_viz_7 = self._env.sim.model.geom_name2id('box_viz_7')
      box_viz_8 = self._env.sim.model.geom_name2id('box_viz_8')
      xarm_viz_indices = 2#[geom_dict[name] for name in geom_dict.keys() if "xarm_viz" in name]
      model = self._env.sim.model

      dr_update_dict = {
        'joint1_damping': model.dof_damping[0],
        'joint2_damping': model.dof_damping[1],
        'joint3_damping': model.dof_damping[2],
        'joint4_damping': model.dof_damping[3],
        'joint5_damping': model.dof_damping[4],
        'joint6_damping': model.dof_damping[5],
        'joint7_damping': model.dof_damping[6],
        'robot_r': model.geom_rgba[xarm_viz_indices, 0],
        'robot_g': model.geom_rgba[xarm_viz_indices, 1],
        'robot_b': model.geom_rgba[xarm_viz_indices, 2],
        'cylinder_r': model.geom_rgba[cylinder_viz, 0],
        'cylinder_g': model.geom_rgba[cylinder_viz, 1],
        'cylinder_b': model.geom_rgba[cylinder_viz, 2],
        'cylinder_mass': model.body_mass[cylinder_body],

        'box1_r': model.geom_rgba[box_viz_1, 0],
        'box1_g': model.geom_rgba[box_viz_1, 1],
        'box1_b': model.geom_rgba[box_viz_1, 2],
        'box2_r': model.geom_rgba[box_viz_2, 0],
        'box2_g': model.geom_rgba[box_viz_2, 1],
        'box2_b': model.geom_rgba[box_viz_2, 2],
        'box3_r': model.geom_rgba[box_viz_3, 0],
        'box3_g': model.geom_rgba[box_viz_3, 1],
        'box3_b': model.geom_rgba[box_viz_3, 2],
        'box4_r': model.geom_rgba[box_viz_4, 0],
        'box4_g': model.geom_rgba[box_viz_4, 1],
        'box4_b': model.geom_rgba[box_viz_4, 2],
        'box5_r': model.geom_rgba[box_viz_5, 0],
        'box5_g': model.geom_rgba[box_viz_5, 1],
        'box5_b': model.geom_rgba[box_viz_5, 2],
        'box6_r': model.geom_rgba[box_viz_6, 0],
        'box6_g': model.geom_rgba[box_viz_6, 1],
        'box6_b': model.geom_rgba[box_viz_6, 2],
        'box7_r': model.geom_rgba[box_viz_7, 0],
        'box7_g': model.geom_rgba[box_viz_7, 1],
        'box7_b': model.geom_rgba[box_viz_7, 2],
        'box8_r': model.geom_rgba[box_viz_8, 0],
        'box8_g': model.geom_rgba[box_viz_8, 1],
        'box8_b': model.geom_rgba[box_viz_8, 2],

        'rope_damping': model.tendon_damping[0],
        'rope_friction': model.tendon_frictionloss[0],
        'rope_stiffness': model.tendon_stiffness[0],

        'lighting': model.light_diffuse[0,0],
      }


      dr_list = []
      for dr_param in self.dr_list:
        dr_list.append(dr_update_dict[dr_param])
      arr = np.array(dr_list)

    else:
      geom_dict = self._env.sim.model._geom_name2id
      stove_collision_indices = [geom_dict[name] for name in geom_dict.keys() if
                                 "stove_collision" in name][0]
      stove_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "stove_viz" in name][0]
      xarm_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "xarm_viz" in name][0]
      xarm_collision_index = [geom_dict[name] for name in geom_dict.keys() if
                                "xarm_collision" in name or "end_effector" in name][0]
      microwave_index = self._env.sim.model.body_name2id('microdoorroot')
      microwave_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_viz" in name][0]
      microwave_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_collision" in name][0]
      cabinet_index = self._env.sim.model.body_name2id('slidelink')
      cabinet_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_viz" in name][0]
      cabinet_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_collision" in name][0]
      data = self._env.sim.data
      model = self._env.sim.model

      dr_update_dict = {
        'joint1_damping': model.dof_damping[0],
        'joint2_damping': model.dof_damping[1],
        'joint3_damping': model.dof_damping[2],
        'joint4_damping': model.dof_damping[3],
        'joint5_damping': model.dof_damping[4],
        'joint6_damping': model.dof_damping[5],
        'joint7_damping': model.dof_damping[6],
        'cabinet_r': model.geom_rgba[cabinet_viz_indices, 0],
        'cabinet_g': model.geom_rgba[cabinet_viz_indices, 1],
        'cabinet_b': model.geom_rgba[cabinet_viz_indices, 2],
        'cabinet_friction': model.geom_friction[cabinet_collision_indices, 0],
        'cabinet_mass': model.body_mass[cabinet_index],

        'knob_mass': model.body_mass[22],
        'lighting': model.light_diffuse[0, 0],

        'microwave_r': model.geom_rgba[microwave_viz_indices, 0],
        'microwave_g': model.geom_rgba[microwave_viz_indices, 1],
        'microwave_b': model.geom_rgba[microwave_viz_indices, 2],
        'microwave_friction': model.geom_friction[microwave_collision_indices, 0],
        'microwave_mass': model.body_mass[microwave_index],
        'robot_r': model.geom_rgba[xarm_viz_indices, 0],
        'robot_g': model.geom_rgba[xarm_viz_indices, 1],
        'robot_b': model.geom_rgba[xarm_viz_indices, 2],
        'stove_r': model.geom_rgba[stove_viz_indices, 0],
        'stove_g': model.geom_rgba[stove_viz_indices, 1],
        'stove_b': model.geom_rgba[stove_viz_indices, 2],
        'stove_friction': model.geom_friction[stove_collision_indices, 0],
      }

      if self.has_kettle:
        kettle_index = self._env.sim.model.body_name2id('kettleroot')
        kettle_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "kettle_viz" in name][0]
        kettle_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "kettle_collision" in name][0]
        dr_update_dict_k = {
        'kettle_r': model.geom_rgba[kettle_viz_indices, 0],
        'kettle_g': model.geom_rgba[kettle_viz_indices, 1],
        'kettle_b': model.geom_rgba[kettle_viz_indices, 2],
        'kettle_friction': model.geom_friction[kettle_collision_indices, 0],
        'kettle_mass': model.body_mass[kettle_index],
        }
        dr_update_dict.update(dr_update_dict_k)

      dr_list = []
      for dr_param in self.dr_list:
        dr_list.append(dr_update_dict[dr_param])
      arr = np.array(dr_list)
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
    if 'open_microwave' in self.task or 'open_cabinet' in self.task:
      self._env.data.set_mocap_quat('mocap', np.array([0.93937271,  0., 0., -0.34289781]))

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
      obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.sim.data.site_xpos[self.end_effector_index])])
      if self.has_kettle:
        obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.sim.data.body_xpos[XPOS_INDICES['kettle']])])
      if self.use_gripper:
        obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.data.qpos[self.arm_njnts])])
    obs['image'] = self.render()
    info['discount'] = 1.0
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 1.0 if done else 0.0

    if not self.early_termination:
      done = False
    return obs, reward, done, info


  def reset(self):


    state_obs = self._env.reset()
    self.setup_task()
    self.apply_dr()

    if 'open_microwave' in self.task or 'open_cabinet' in self.task:
      self._env.data.set_mocap_quat('mocap', np.array([0.93937271, 0., 0., -0.34289781]))
      # Make the end-effector horizontal
      for _ in range(2000):
        self._env.sim.step()


    obs = {}
    obs['state'] = self.goal
    if self.outer_loop_version == 1:
      obs['sim_params'] = self.sim_params
    if self.use_state:
      obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.sim.data.site_xpos[self.end_effector_index])])
      if self.has_kettle:
        obs['state'] = np.concatenate([obs['state'], np.squeeze(self._env.sim.data.body_xpos[XPOS_INDICES['kettle']])])
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
  def __init__(self, name, size=(64, 64), mean_only=False, early_termination=False, dr_list=[], simple_randomization=False, dr_shape=None, outer_loop_version=0,
               real_world=False, dr=None, use_state=False, azimuth=-30, distance=2, elevation=-20, use_depth=False):
    from environments.metaworld.metaworld import ML1
    import random
    self._ml1 = ML1(name + "-v1")
    self._env = self._ml1.train_classes[name + "-v1"]()
    self.name = name
    task = random.choice(self._ml1.train_tasks)
    self._env.set_task(task)

    self._size = size

    self.mean_only = mean_only
    self.early_termination = early_termination
    self.dr_list = dr_list
    self.simple_randomization = simple_randomization
    self.dr_shape = dr_shape
    self.outer_loop_version = outer_loop_version
    self.real_world = real_world
    self.use_state = use_state
    self.dr = dr
    self.use_depth = use_depth

    if self._env.viewer is None:
      from mujoco_py import MjRenderContextOffscreen

      self.viewer = MjRenderContextOffscreen(self._env.sim, device_id=-1)
      self.viewer.cam.elevation = elevation
      self.viewer.cam.azimuth = azimuth
      self.viewer.cam.distance = distance

    self.apply_dr()

  def update_dr_param(self, param, param_name, eps=1e-3, indices=None):
    if param_name in self.dr:
      if self.mean_only:
        mean = self.dr[param_name]
        range = max(0.1 * mean, eps) #TODO: tune this?
      else:
        mean, range = self.dr[param_name]
        range = max(range, eps)
      new_value = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
      if indices is None:
        param[:] = new_value
      else:
        try:
          for i in indices:
            param[i:i+1] = new_value
        except:
          param[indices:indices+1] = new_value

      if self.mean_only:
        self.sim_params += [mean]
      else:
        self.sim_params += [mean, range]

  def apply_dr(self):
    self.sim_params = []
    if self.dr is None or self.real_world:
      if self.outer_loop_version == 1:
        self.sim_params = np.zeros(self.dr_shape)
      return

    model = self._env.sim.model
    geom_dict = model._geom_name2id
    body_dict = model._body_name2id
    robot_geom = [
      geom_dict['right_arm_base_link_geom'],
      geom_dict['right_l0_geom'],
      geom_dict['right_l1_geom'],
      geom_dict['right_l2_geom'],
      geom_dict['right_l3_geom'],
      geom_dict['right_l4_geom'],
      geom_dict['right_l5_geom'],
      geom_dict['right_l6_geom'],
      geom_dict['right_hand_geom'],
      geom_dict['head_geom'],
    ]
    table_geom = geom_dict['tableTop']

    dr_update_dict_common = {
      # Table
      "table_friction": (model.geom_friction[table_geom: table_geom + 1], None),
      "table_r": (model.geom_rgba[table_geom: table_geom + 1, 0], None),
      "table_g": (model.geom_rgba[table_geom: table_geom + 1, 1], None),
      "table_b": (model.geom_rgba[table_geom: table_geom + 1, 2], None),

      # Robot
      'robot_r': (model.geom_rgba[:, 0], robot_geom),
      'robot_g': (model.geom_rgba[:, 1], robot_geom),
      'robot_b': (model.geom_rgba[:, 2], robot_geom),
      'robot_friction': (model.geom_friction[:, 0], robot_geom),
    }

    if self.name in ['stick-pull', 'stick-push']:
      stick_body = body_dict['stick']
      stick_geom = geom_dict['stick_geom_1']

      object_body = body_dict['object']
      object_geom_body = [
        geom_dict['object_geom_1'],
      ]
      object_geom_handle = [
        geom_dict['object_geom_handle_1'],
        geom_dict['object_geom_handle_2'],
        geom_dict['handle']
      ]
      object_geom = object_geom_body + object_geom_handle

      dr_update_dict = {
          # Stick
          "stick_mass": (model.body_mass[stick_body: stick_body + 1], None),
          "stick_friction": (model.geom_friction[stick_geom: stick_geom + 1, 0], None),
          "stick_r": (model.geom_rgba[stick_geom: stick_geom + 1, 0], None),
          "stick_g": (model.geom_rgba[stick_geom: stick_geom + 1, 1], None),
          "stick_b": (model.geom_rgba[stick_geom: stick_geom + 1, 2], None),

          # Object
          "object_mass": (model.body_mass[object_body: object_body + 1], None),
          "object_friction": (model.geom_friction[:, 0], object_geom),
          "object_body_r": (model.geom_rgba[:, 0], object_geom_body),
          "object_body_g": (model.geom_rgba[:, 1], object_geom_body),
          "object_body_b": (model.geom_rgba[:, 2], object_geom_body),
          "object_handle_r": (model.geom_rgba[:, 0], object_geom_handle),
          "object_handle_g": (model.geom_rgba[:, 1], object_geom_handle),
          "object_handle_b": (model.geom_rgba[:, 2], object_geom_handle),
        }
      dr_update_dict.update(dr_update_dict_common)


      for dr_param in self.dr_list:
        arr, indices = dr_update_dict[dr_param]
        print(dr_param, arr, indices)
        self.update_dr_param(arr, dr_param, indices=indices)

    elif 'basketball' in self.name:
      basket_goal_geom = [
        geom_dict['handle'],
        geom_dict['basket_goal_geom_1'],
        geom_dict['basket_goal_geom_2'],
      ]
      backboard_geom = [
        geom_dict['basket_goal'],
      ]
      basket_geom = basket_goal_geom + backboard_geom

      object_body = body_dict['obj']
      object_geom = [geom_dict['objGeom']]

      dr_update_dict = {
        # Stick
        "basket_friction": (model.geom_rgba[:, 2], basket_geom),
        "basket_goal_r": (model.geom_rgba[:, 0], basket_goal_geom),
        "basket_goal_g": (model.geom_rgba[:, 1], basket_goal_geom),
        "basket_goal_b": (model.geom_rgba[:, 2], basket_goal_geom),
        "backboard_r": (model.geom_rgba[:, 0], backboard_geom),
        "backboard_g": (model.geom_rgba[:, 1], backboard_geom),
        "backboard_b": (model.geom_rgba[:, 2], backboard_geom),

        # Object
        "object_mass": (model.body_mass[object_body: object_body + 1], None),
        "object_friction": (model.geom_friction[:, 0], object_geom),
        "object_r": (model.geom_rgba[:, 0], object_geom),
        "object_g": (model.geom_rgba[:, 1], object_geom),
        "object_b": (model.geom_rgba[:, 2], object_geom),
      }
      dr_update_dict.update(dr_update_dict_common)

      for dr_param in self.dr_list:
        arr, indices = dr_update_dict[dr_param]
        self.update_dr_param(arr, dr_param, indices=indices)

    else:
      raise NotImplementedError


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
    time_out = self._env.curr_path_length == self._env.max_path_length
    done = done or time_out
    obs = {}
    obs['state'] = self._env._get_pos_goal()
    if self.use_state:
      obs['state'] = np.concatenate([obs['state'], state_obs[:3]])  # Only include robot state (endeffector pos)
    obs['image'] = self.render()
    info['discount'] = 1.0
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 1.0 if info['success'] else 0.0

    if not self.early_termination:
      done = False

    return obs, reward, done, info

  def get_dr(self):
    if self.simple_randomization:
      if 'stick-pull' or 'stick-push' in self.name:
        cylinder_body = self._env.sim.model.body_name2id('cylinder')
        return np.array([self._env.sim.model.body_mass[cylinder_body]])
      elif 'basketball' in self.name:
        microwave_index = self._env.sim.model.body_name2id('microdoorroot')
        return np.array([self._env.sim.model.body_mass[microwave_index]])
      else:
        raise NotImplementedError

    model = self._env.sim.model
    geom_dict = model._geom_name2id
    body_dict = model._body_name2id
    robot_geom = [
      geom_dict['right_arm_base_link_geom'],
      geom_dict['right_l0_geom'],
      geom_dict['right_l1_geom'],
      geom_dict['right_l2_geom'],
      geom_dict['right_l3_geom'],
      geom_dict['right_l4_geom'],
      geom_dict['right_l5_geom'],
      geom_dict['right_l6_geom'],
      geom_dict['right_hand_geom'],
      geom_dict['head_geom'],
    ]
    table_geom = geom_dict['tableTop']

    dr_update_dict_common = {
      # Table
      "table_friction": model.geom_friction[table_geom, 0],
      "table_r": model.geom_rgba[table_geom, 0],
      "table_g": model.geom_rgba[table_geom, 1],
      "table_b": model.geom_rgba[table_geom, 2],

      # Robot
      'robot_r': model.geom_rgba[robot_geom[0], 0],
      'robot_g': model.geom_rgba[robot_geom[0], 1],
      'robot_b': model.geom_rgba[robot_geom[0], 2],
      'robot_friction': model.geom_friction[robot_geom[0], 0],
    }

    if self.name in ['stick-pull', 'stick-push']:
      model = self._env.sim.model
      geom_dict = model._geom_name2id
      body_dict = model._body_name2id

      stick_body = body_dict['stick']
      stick_geom = geom_dict['stick_geom_1']

      object_body = body_dict['object']
      object_geom_body = [
        geom_dict['object_geom_1'],
      ]
      object_geom_handle = [
        geom_dict['object_geom_handle_1'],
        geom_dict['object_geom_handle_2'],
        geom_dict['handle']
      ]
      object_geom = object_geom_body + object_geom_handle

      dr_update_dict = {
        # Stick
        "stick_mass": model.body_mass[stick_body],
        "stick_friction": model.geom_friction[stick_geom, 0],
        "stick_r": model.geom_rgba[stick_geom, 0],
        "stick_g": model.geom_rgba[stick_geom, 1],
        "stick_b": model.geom_rgba[stick_geom, 2],

        # Object
        "object_mass": model.body_mass[object_body],
        "object_friction": model.geom_friction[object_geom[0], 0],
        "object_body_r": model.geom_rgba[object_geom_body[0], 0],
        "object_body_g": model.geom_rgba[object_geom_body[0], 1],
        "object_body_b": model.geom_rgba[object_geom_body[0], 2],
        "object_handle_r": model.geom_rgba[object_geom_handle[0], 0],
        "object_handle_g": model.geom_rgba[object_geom_handle[0], 1],
        "object_handle_b": model.geom_rgba[object_geom_handle[0], 2],
      }
      dr_update_dict.update(dr_update_dict_common)

      dr_list = []
      for dr_param in self.dr_list:
        dr_list.append(dr_update_dict[dr_param])
      arr = np.array(dr_list)
    elif 'basketball' in self.name:
      basket_goal_geom = [
        geom_dict['handle'],
        geom_dict['basket_goal_geom_1'],
        geom_dict['basket_goal_geom_2'],
      ]
      backboard_geom = [
        geom_dict['basket_goal'],
      ]
      basket_geom = basket_goal_geom + backboard_geom

      object_body = body_dict['obj']
      object_geom = [geom_dict['objGeom']]

      dr_update_dict = {
        # Stick
        "basket_friction": model.geom_rgba[basket_geom[0], 2],
        "basket_goal_r": model.geom_rgba[basket_goal_geom[0], 0],
        "basket_goal_g": model.geom_rgba[basket_goal_geom[0], 1],
        "basket_goal_b": model.geom_rgba[basket_goal_geom[0], 2],
        "backboard_r": model.geom_rgba[basket_goal_geom[0], 0],
        "backboard_g": model.geom_rgba[basket_goal_geom[0], 1],
        "backboard_b": model.geom_rgba[basket_goal_geom[0], 2],

        # Object
        "object_mass": model.body_mass[object_body],
        "object_friction": model.geom_friction[object_geom[0], 0],
        "object_r": model.geom_rgba[object_geom[0], 0],
        "object_g": model.geom_rgba[object_geom[0], 1],
        "object_b": model.geom_rgba[object_geom[0], 2],
      }
      dr_update_dict.update(dr_update_dict_common)

      dr_list = []
      for dr_param in self.dr_list:
        dr_list.append(dr_update_dict[dr_param])
      arr = np.array(dr_list)

    arr = arr.astype(np.float32)
    return arr

  def reset(self):
    state_obs = self._env.reset()
    self.apply_dr()

    obs = {}
    obs['state'] = self._env._get_pos_goal()
    if self.use_state:
      obs['state'] = np.concatenate([obs['state'], state_obs[:3]])  # Only include robot state (endeffector pos)
    obs['image'] = self.render()
    obs['real_world'] = 1.0 if self.real_world else 0.0
    obs['dr_params'] = self.get_dr()
    obs['success'] = 0.0
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    width, height = self._size

    if self.viewer is not None:
      self.viewer.update_sim(self._env.sim)
      self.viewer.render(*self._size)

      data = self.viewer.read_pixels(*self._size, depth=self.use_depth)
      if self.use_depth:
        img, depth = data
        img = img[::-1]
        depth = depth[::-1] * 255
        depth = depth[..., None]
        return np.concatenate([img, depth], axis=-1).astype(int)

      return data[::-1]

    return self._env.sim.render(mode='offscreen', width=width, height=height)

class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None, real_world=False, sparse_reward=True, dr=None, use_state=False,
                                     simple_randomization=False, dr_shape=None, outer_loop_type=0, dr_list=[],
               mean_only=False):

    self.task = name
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
    self.dr_list = dr_list
    self.mean_only = mean_only

    self.apply_dr()

  def update_dr_param(self, param, param_name, eps=1e-3, indices=None):
    if param_name in self.dr:
      if self.mean_only:
        mean = self.dr[param_name]
        range = max(0.1 * mean, eps) #TODO: tune this?
      else:
        mean, range = self.dr[param_name]
        range = max(range, eps)
      new_value = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
      if indices is None:
        param[:] = new_value
      else:
        try:
          for i in indices:
            param[i:i+1] = new_value
        except:
          param[indices:indices+1] = new_value

      if self.mean_only:
        self.sim_params += [mean]
      else:
        self.sim_params += [mean, range]


  def apply_dr(self):
    self.sim_params = []
    if self.dr is None or self.real_world:
      if self.outer_loop_version == 1:
        self.sim_params = np.zeros(self.dr_shape)
      return  # TODO: start using XPOS_INDICES or equivalent for joints.

    model = self._env.physics.model
    if 'cup_catch' in self.task:
      dr_update_dict = {
        "cup_mass": model.body_mass[1:2],
        "ball_mass": model.body_mass[2:3],
        "cup_damping": model.dof_damping[0:2],
        "ball_damping": model.dof_damping[2:4],
        "actuator_gain": model.actuator_gainprm[:, 0],
        "cup_r": model.geom_rgba[0:6, 0],
        "cup_g": model.geom_rgba[0:6, 1],
        "cup_b": model.geom_rgba[0:6, 2],
        "ball_r": model.geom_rgba[6:7, 0],
        "ball_g": model.geom_rgba[6:7, 1],
        "ball_b": model.geom_rgba[6:7, 2],
      }
    elif "walker" in self.task:
      dr_update_dict = {
        "torso_mass": model.body_mass[1:2],
        "right_thigh_mass": model.body_mass[2:3],
        "right_leg_mass": model.body_mass[3:4],
        "right_foot_mass": model.body_mass[4:5],
        "left_thigh_mass": model.body_mass[5:6],
        "left_leg_mass": model.body_mass[6:7],
        "left_foot_mass": model.body_mass[7:8],
        "right_hip": model.dof_damping[3:4],
        "right_knee": model.dof_damping[4:5],
        "right_ankle": model.dof_damping[5:6],
        "left_hip": model.dof_damping[6:7],
        "left_knee": model.dof_damping[7:8],
        "left_ankle": model.dof_damping[8:9],
        "ground_r": model.geom_rgba[0:1, 0],
        "ground_g": model.geom_rgba[0:1, 1],
        "ground_b": model.geom_rgba[0:1, 2],
        "body_r": model.geom_rgba[1:8, 0],
        "body_g": model.geom_rgba[1:8, 1],
        "body_b": model.geom_rgba[1:8, 2],
      }
    elif "cheetah" in self.task:
      dr_update_dict = {
        "torso_mass": model.body_mass[1:2],
        "bthigh_mass": model.body_mass[2:3],
        "bshin_mass": model.body_mass[3:4],
        "bfoot_mass": model.body_mass[4:5],
        "fthigh_mass": model.body_mass[5:6],
        "fshin_mass": model.body_mass[6:7],
        "ffoot_mass": model.body_mass[7:8],
        "bthigh_damping": model.dof_damping[3:4],
        "bshin_damping": model.dof_damping[4:5],
        "bfoot_damping": model.dof_damping[5:6],
        "fthigh_damping": model.dof_damping[6:7],
        "fshin_damping": model.dof_damping[7:8],
        "ffoot_damping": model.dof_damping[8:9],
        "ground_r": model.geom_rgba[0:1, 0],
        "ground_g": model.geom_rgba[0:1, 1],
        "ground_b": model.geom_rgba[0:1, 2],
        "body_r": model.geom_rgba[1:9, 0],
        "body_g": model.geom_rgba[1:9, 1],
        "body_b": model.geom_rgba[1:9, 2],
      }
    elif "finger" in self.task:
      dr_update_dict = {
        "proximal_mass": model.body_mass[0:1],
        "distal_mass": model.body_mass[1:2],
        "spinner_mass": model.body_mass[2:3],
        "proximal_damping": model.dof_damping[0:1],
        "distal_damping": model.dof_damping[1:2],
        "hinge_damping": model.dof_damping[2:3],
        "ground_r": model.geom_rgba[0:1, 0],
        "ground_g": model.geom_rgba[0:1, 1],
        "ground_b": model.geom_rgba[0:1, 2],
        "finger_r": model.geom_rgba[2:4, 0],
        "finger_g": model.geom_rgba[2:4, 1],
        "finger_b": model.geom_rgba[2:4, 2],
        "hotdog_r": model.geom_rgba[5:7, 0],
        "hotdog_g": model.geom_rgba[5:7, 1],
        "hotdog_b": model.geom_rgba[5:7, 2],
      }
    # Actually Update
    for dr_param in self.dr_list:
      arr = dr_update_dict[dr_param]
      self.update_dr_param(arr, dr_param)


  def get_dr(self):
    model = self._env.physics.model
    if "cup_catch" in self.task:
      dr_update_dict = {
        "cup_mass": model.body_mass[1],
        "ball_mass": model.body_mass[2],
        "cup_damping": model.dof_damping[0],
        "ball_damping": model.dof_damping[2],
        "actuator_gain": model.actuator_gainprm[0, 0],
        "cup_r": model.geom_rgba[0, 0],
        "cup_g": model.geom_rgba[0, 1],
        "cup_b": model.geom_rgba[0, 2],
        "ball_r": model.geom_rgba[6, 0],
        "ball_g": model.geom_rgba[6, 1],
        "ball_b": model.geom_rgba[6, 2],
      }
    elif "walker" in self.task:
      dr_update_dict = {
        "torso_mass": model.body_mass[1],
        "right_thigh_mass": model.body_mass[2],
        "right_leg_mass": model.body_mass[3],
        "right_foot_mass": model.body_mass[4],
        "left_thigh_mass": model.body_mass[5],
        "left_leg_mass": model.body_mass[6],
        "left_foot_mass": model.body_mass[7],
        "right_hip": model.dof_damping[3],
        "right_knee": model.dof_damping[4],
        "right_ankle": model.dof_damping[5],
        "left_hip": model.dof_damping[6],
        "left_knee": model.dof_damping[7],
        "left_ankle": model.dof_damping[8],
        "ground_r": model.geom_rgba[0, 0],
        "ground_g": model.geom_rgba[0, 1],
        "ground_b": model.geom_rgba[0, 2],
        "body_r": model.geom_rgba[1, 0],
        "body_g": model.geom_rgba[1, 1],
        "body_b": model.geom_rgba[1, 2],
      }
    elif "cheetah" in self.task:
      dr_update_dict = {
        "torso_mass": model.body_mass[1],
        "bthigh_mass": model.body_mass[2],
        "bshin_mass": model.body_mass[3],
        "bfoot_mass": model.body_mass[4],
        "fthigh_mass": model.body_mass[5],
        "fshin_mass": model.body_mass[6],
        "ffoot_mass": model.body_mass[7],
        "bthigh_damping": model.dof_damping[3],
        "bshin_damping": model.dof_damping[4],
        "bfoot_damping": model.dof_damping[5],
        "fthigh_damping": model.dof_damping[6],
        "fshin_damping": model.dof_damping[7],
        "ffoot_damping": model.dof_damping[8],
        "ground_r": model.geom_rgba[0, 0],
        "ground_g": model.geom_rgba[0, 1],
        "ground_b": model.geom_rgba[0, 2],
        "body_r": model.geom_rgba[1, 0],
        "body_g": model.geom_rgba[1, 1],
        "body_b": model.geom_rgba[1, 2],
      }
    elif "finger" in self.task:
      dr_update_dict = {
        "proximal_mass": model.body_mass[0],
        "distal_mass": model.body_mass[1],
        "spinner_mass": model.body_mass[2],
        "proximal_damping": model.dof_damping[0],
        "distal_damping": model.dof_damping[1],
        "hinge_damping": model.dof_damping[2],
        "ground_r": model.geom_rgba[0, 0],
        "ground_g": model.geom_rgba[0, 1],
        "ground_b": model.geom_rgba[0, 2],
        "finger_r": model.geom_rgba[2, 0],
        "finger_g": model.geom_rgba[2, 1],
        "finger_b": model.geom_rgba[2, 2],
        "hotdog_r": model.geom_rgba[5, 0],
        "hotdog_g": model.geom_rgba[5, 1],
        "hotdog_b": model.geom_rgba[5, 2],
      }

    dr_list = []
    for dr_param in self.dr_list:
      dr_list.append(dr_update_dict[dr_param])
    arr = np.array(dr_list)

    arr = arr.astype(np.float32)
    return arr


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

  # def get_dr(self):
  #   if self.simple_randomization:
  #     return np.array([self._env.physics.model.body_mass[2]])
  #   return np.array([
  #     self._env.physics.model.actuator_gainprm[0, 0],
  #     self._env.physics.model.body_mass[2],
  #     # self._env.physics.model.geom_rbound[-1],
  #     self._env.physics.model.dof_damping[0],
  #     self._env.physics.model.geom_friction[0, 0],
  #     # self._env.physics.model.tendon_length0[0],
  #     # self._env.physics.model.tendon_stiffness[0],
  #   ])

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
