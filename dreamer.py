import argparse
import collections
import copy
import functools
import json
import os
import pathlib
import sys
import time
import shutil
import psutil
import cv2
import pickle as pkl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools


# FB Cluster SBatch ======================================================
import signal
MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False


def SIGTERMHandler(a, b):
    print('received sigterm')
    pass


def signalHandler(a, b):
    global SIGNAL_RECEIVED
    print('Signal received', a, time.time(), flush=True)
    SIGNAL_RECEIVED = True
    trigger_job_requeue()
    return


def trigger_job_requeue():
    ''' Submit a new job to resume from checkpoint.
    '''
    if os.environ['SLURM_PROCID'] == '0' and \
       os.getpid() == MAIN_PID:
        ''' BE AWARE OF subprocesses that your program spawns.
        Only the main process on slurm procID = 0 resubmits the job.
        In pytorch imagenet example, by default it spawns 4
        (specified in -j) subprocesses for data loading process,
        both parent process and child processes will receive the signal.
        Please only submit the job in main process,
        otherwise the job queue will be filled up exponentially.
        Command below can be used to check the pid of running processes.
        print('pid: ', os.getpid(), ' ppid: ', os.getppid(), flush=True)
        '''
        print('time is up, back to slurm queue', flush=True)
        command = 'scontrol requeue ' + os.environ['SLURM_JOB_ID']
        print(command)
        if os.system(command):
            raise RuntimeError('requeue failed')
        print('New job submitted to the queue', flush=True)
    exit(0)


# Install signal handler
signal.signal(signal.SIGUSR1, signalHandler)
signal.signal(signal.SIGTERM, SIGTERMHandler)
print('Signal handler installed', flush=True)
# ========================================================================


def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.')
  config.seed = 0
  config.steps = 2e6
  config.eval_every = 1e4
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  # Environment.
  config.task = 'dmc_cup_catch'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 2
  config.time_limit = 200
  config.prefill = 5000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  # Model.
  config.deter_size = 200
  config.stoch_size = 30
  config.num_units = 400
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.dr_lr = 5e-3
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.99
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  config.id = 'debug'
  config.use_state = 'None'  # Options are 'None', 'robot', or 'all'
  config.use_img = True #no image only added for metaworld
  config.num_dr_grad_steps = 100
  config.control_version = 'mocap_ik'
  config.generate_videos = False  # If true, it doesn't train; just generates videos
  config.step_repeat = 50
  config.bounds = 'stove_area'
  config.step_size = 0.01

  # Sim2real transfer
  config.real_world_prob = -1   # fraction of samples trained on which are from the real world (probably involves oversampling real-world samples)
  config.sample_real_every = 2  # How often we should sample from the real world
  config.num_real_world = 1  # Each time we sample from the real world, how many trajectories should we sample?
  config.simple_randomization = False

  # these values are for testing dmc_cup_catch
  config.mass_mean = 0.2
  config.mass_range = 0.01
  config.mean_scale = 0.67
  config.range_scale = 0.33
  config.mean_only = True
  config.anneal_range_scale = 1 #if > 0, start at anneal_range_scale*mean and anneal to 0.1*mean over total_steps
  config.predict_val = True
  config.range_only = False

  config.outer_loop_version = 0  # 0= no outer loop, 1 = regression,f 2 = conditioning
  config.alpha = 0.3
  config.sim_params_size = 0
  config.buffer_size = 0
  config.update_target_every = 1#00
  config.early_termination = False
  config.sim_param_regularization = .0001
  config.use_depth = False
  config.ol1_episodes = 1

  config.random_crop = False
  config.initial_randomization_steps = 3
  config.last_param_pred_only = False
  config.individual_loss_scale = True #True for scaling sim_param loss only, False for scaling sim_param and image+reward loss
  config.sim_params_loss_scale = 1e4
  config.binary_prediction = False

  # Dataset Generation
  config.generate_dataset = False
  config.num_real_episodes = 100
  config.num_sim_episodes = 10000
  config.num_dr_steps = 1 if config.range_only else 10
  config.starting_mean_scale = 5
  config.starting_range_scale = .5
  config.ending_mean_scale = 1
  config.ending_range_scale = .5
  config.minimal = False

  # Using offline dataset
  config.use_offline_dataset = False
  config.datadir = "temp"
  config.grayscale = False



  return config

def config_dr(config):
  dr_option = config.dr_option
  if 'kitchen' in config.task:
    if config.simple_randomization:
      if 'rope' in config.task:
        config.real_dr_params = {
          "cylinder_mass": .5
        }
        config.dr = {  # (mean, range)
          "cylinder_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ["cylinder_mass"]
        config.sim_params_size = 1
      elif 'open_microwave' in config.task:
        config.real_dr_params = {
          "microwave_mass": .26
        }
        config.dr = {  # (mean, range)
          "microwave_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ['microwave_mass']
        config.sim_params_size = 1
      elif 'open_cabinet' in config.task:
        config.real_dr_params = {
          "cabinet_mass": 3.4
        }
        config.dr = {  # (mean, range)
          "cabinet_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ['cabinet_mass']
        config.sim_params_size = 1
      else:
        config.real_dr_params = {
          "kettle_mass": 1.08
        }
        config.dr = {  # (mean, range)
          "kettle_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ['kettle_mass']
        config.sim_params_size = 1
    else:
      if 'rope' in config.task:
        config.real_dr_params = {
          "joint1_damping": 10,
          "joint2_damping": 10,
          "joint3_damping": 5,
          "joint4_damping": 5,
          "joint5_damping": 5,
          "joint6_damping": 2,
          "joint7_damping": 2,
          "robot_b": 0.95,
          "robot_g": 0.95,
          "robot_r": 0.95,
          "cylinder_b": .2,
          "cylinder_g": .2,
          "cylinder_r": 1.,
          "cylinder_mass": 0.5,
          "box1_r": .2,
          "box1_g": 1,
          "box1_b": .2,
          "box2_r": .2,
          "box2_g": 1,
          "box2_b": .2,
          "box3_r": .2,
          "box3_g": 1,
          "box3_b": .2,
          "box4_r": .2,
          "box4_g": 1,
          "box4_b": .2,
          "box5_r": .2,
          "box5_g": 1,
          "box5_b": .2,
          "box6_r": .2,
          "box6_g": 1,
          "box6_b": .2,
          "box7_r": .2,
          "box7_g": 1,
          "box7_b": .2,
          "box8_r": .2,
          "box8_g": 1,
          "box8_b": .2,
          "rope_damping": 0,
          "rope_friction": 0,
          "rope_stiffness": 0,
          "lighting": 0.3
        }
        if dr_option == 'partial_dr':
          config.real_dr_list = ["cylinder_mass", "rope_damping", "rope_friction", "rope_stiffness"]
        elif dr_option == 'all_dr':
          config.real_dr_list = [
            "joint1_damping", "joint2_damping", "joint3_damping", "joint4_damping", "joint5_damping",  "joint6_damping",
            "joint7_damping",  "robot_b",  "robot_g", "robot_r", "cylinder_b", "cylinder_g",
            "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b", "box2_r", "box2_g", "box2_b", "box3_r",
            "box3_g", "box3_b", "box4_r",  "box4_g",  "box4_b", "box5_r", "box5_g", "box5_b", "box6_r", "box6_g",
            "box6_b", "box7_r",  "box7_g", "box7_b", "box8_r", "box8_g", "box8_b", "rope_damping", "rope_friction",
            "rope_stiffness", "lighting",
          ]
        elif dr_option == 'nonconflicting_dr':
          config.real_dr_list = [
            "joint7_damping",  "robot_b",  "robot_g", "robot_r", "cylinder_b", "cylinder_g",
            "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b", "box2_r", "box2_g", "box2_b", "box3_r",
            "box3_g", "box3_b", "box4_r",  "box4_g",  "box4_b", "box5_r", "box5_g", "box5_b", "box6_r", "box6_g",
            "box6_b", "box7_r",  "box7_g", "box7_b", "box8_r", "box8_g", "box8_b", "rope_damping", "lighting",
          ]

      else:
        config.real_dr_params = {
          "cabinet_b": 0.5,
          "cabinet_friction": 1,
          "cabinet_g": 0.5,
          "cabinet_mass": 3.4,
          "cabinet_r": 0.5,
          "joint1_damping": 10,
          "joint2_damping": 10,
          "joint3_damping": 5,
          "joint4_damping": 5,
          "joint5_damping": 5,
          "joint6_damping": 2,
          "joint7_damping": 2,
          "kettle_b": 0.5,
          "kettle_friction": 1.0,
          "kettle_g": 0.5,
          "kettle_mass": 1.08,
          "kettle_r": 0.5,
          "knob_mass": 0.02,
          "lighting": 0.3,
          "microwave_b": 0.5,
          "microwave_friction": 1,
          "microwave_g": 0.5,
          "microwave_mass": .26,
          "microwave_r": 0.5,
          "robot_b": 0.92,
          "robot_g": .99,
          "robot_r": 0.95,
          "stove_b": 0.5,
          "stove_friction": 1.,
          "stove_g": 0.5,
          "stove_r": 0.5,
        }
        if dr_option == 'partial_dr':
          config.real_dr_list = [
            "cabinet_b", "cabinet_g", "cabinet_mass", "cabinet_r", "joint7_damping", "kettle_b",
            "kettle_g", "kettle_mass", "kettle_r", "lighting", "microwave_b", "kettle_friction",
            "microwave_g", "microwave_mass", "microwave_r", "robot_b", "robot_g", "robot_r", "stove_b", "stove_g", "stove_r",
          ]
        elif dr_option == 'all_dr':
          config.real_dr_list = [
            "cabinet_b", "cabinet_friction", "cabinet_g", "cabinet_mass", "cabinet_r", "joint1_damping", "joint2_damping",
            "joint3_damping", "joint4_damping", "joint5_damping", "joint6_damping", "joint7_damping", "kettle_b", "kettle_friction",
            "kettle_g", "kettle_mass", "kettle_r",  "knob_mass", "lighting", "microwave_b", "microwave_friction",
            "microwave_g",  "microwave_mass", "microwave_r", "robot_b", "robot_g",  "robot_r", "stove_b",
            "stove_friction",  "stove_g", "stove_r",
          ]
        elif dr_option == 'dynamics_dr':
          config.real_dr_list = ["cabinet_mass", "joint7_damping", "kettle_mass", "kettle_friction"]
        elif dr_option == 'friction_dr':
          config.real_dr_list = ["kettle_friction", "cabinet_friction"]
        elif dr_option == 'dynamics_nonconflicting_dr':
          config.real_dr_list = ["cabinet_mass", "joint7_damping", "kettle_mass"]
        elif dr_option == 'nonconflicting_dr':
          config.real_dr_list = ["cabinet_mass", "joint7_damping", "kettle_mass", "cabinet_b", "cabinet_g", "cabinet_r",
                                 "kettle_b", "kettle_g", "kettle_r", "lighting", "microwave_b", "microwave_g",
                                 "microwave_r", "robot_b", "robot_g",  "robot_r", "stove_b",  "stove_g", "stove_r"]
        elif dr_option == 'visual':
          config.real_dr_list = ["stove_r"]
        elif dr_option == 'mass':
          config.real_dr_list = ["kettle_mass" if "kettle" in config.task else "cabinet_mass"]
        elif dr_option == 'friction':
          config.real_dr_list = ["kettle_friction" if "kettle" in config.task else "cabinet_friction"]

        if 'slide' in config.task:
          config.real_dr_params['stove_friction'] = 1e-3
          config.real_dr_params['kettle_friction'] = 1e-3


        # Remove kettle-related d-r for the microwave task, which has no kettle present.
        if 'open_microwave' in config.task:
          for k in list(config.real_dr_params.keys()):
            if 'kettle' in k:
              del config.real_dr_params[k]


      config.sim_params_size = len(config.real_dr_list)
      mean_scale = config.mean_scale
      range_scale = config.range_scale
      config.dr = {}  # (mean, range)
      for key, real_val in config.real_dr_params.items():
        if real_val == 0:
          real_val = 5e-2
        config.dr[key] = (real_val * mean_scale, real_val * range_scale)

        #Keep mean only
    if config.mean_only and config.dr is not None:
      dr = {}
      for key, vals in config.dr.items():
        dr[key] = vals[0] #only keep mean
      config.dr = dr

  elif "metaworld" in config.task:
    if config.simple_randomization:
      if 'basketball' in config.task:
        config.real_dr_params = {
          "object_mass": .01
        }
        config.dr = {  # (mean, range)
          "object_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ["object_mass"]
        config.sim_params_size = 1
      elif 'stick' in config.task:
        if 'basketball' in config.task:
          config.real_dr_params = {
            "object_mass": .128
          }
          config.dr = {  # (mean, range)
            "object_mass": (config.mass_mean, config.mass_range)
          }
          config.real_dr_list = ["object_mass"]
          config.sim_params_size = 1
    elif config.dr_option == 'all_dr':
      real_dr_joint = {
        "table_friction": 2.,
        "table_r": .6,
        "table_g": .6,
        "table_b": .5,
        "robot_friction": 1.,
        "robot_r": .5,
        "robot_g": .1,
        "robot_b": .1,
      }
      if 'basketball' in config.task:
        config.real_dr_params = {
          "basket_friction": .5,
          "basket_goal_r": .5,
          "basket_goal_g": .5,
          "basket_goal_b": .5,
          "backboard_r": .5,
          "backboard_g": .5,
          "backboard_b": .5,
          "object_mass": .01,
          "object_friction": 1.,
          "object_r": 0.,
          "object_g": 0.,
          "object_b": 0.,
        }
        config.real_dr_params.update(real_dr_joint)
        config.real_dr_list = list(config.real_dr_params.keys())
      elif 'stick' in config.task:
        config.real_dr_params = {
          "stick_mass": 1.,
          "stick_friction": 1.,
          "stick_r": 1.,
          "stick_g": .3,
          "stick_b": .3,
          "object_mass": .128,
          "object_friction": 1.,
          "object_body_r": 0.,
          "object_body_g": 0.,
          "object_body_b": 1.,
          "object_handle_r": 0,
          "object_handle_g": 0,
          "object_handle_b": 0,
        }
        config.real_dr_params.update(real_dr_joint)
        config.real_dr_list = list(config.real_dr_params.keys())
      config.sim_params_size = len(config.real_dr_list)
      mean_scale = config.mean_scale
      range_scale = config.range_scale
      config.dr = {}  # (mean, range)
      for key, real_val in config.real_dr_params.items():
        if real_val == 0:
          real_val = 5e-2
        if config.mean_only:
          config.dr[key] = real_val * mean_scale
        else:
          config.dr[key] = (real_val * mean_scale, real_val * range_scale)
    else:
      config.dr = {}
      config.real_dr_params = {}
      config.dr_list = []
  elif "dmc" in config.task:
    if "cup_catch" in config.task:
      real_dr_values = {
        "cup_mass": .0625,
        "ball_mass": .0654,
        "cup_damping": 3.,
        "ball_damping": 3.,
        "actuator_gain": 1.,
        "cup_r": .5,
        "cup_g": .5,
        "cup_b": .5,
        "ball_r": .5,
        "ball_g": .5,
        "ball_b": .5,
      }
      if dr_option == 'all_dr':
        config.real_dr_list = list(real_dr_values.keys())
      elif dr_option == 'nonconflicting_dr':
        config.real_dr_list = [
          "cup_mass", "ball_mass", "cup_r", "cup_g", "cup_b", "ball_r", "ball_g", "ball_b",
        ]
    elif "walker" in config.task:
      real_dr_values = {
        "torso_mass": 10.3,
        "right_thigh_mass": 3.93,
        "right_leg_mass": 2.71,
        "right_foot_mass": 1.96,
        "left_thigh_mass": 3.93,
        "left_leg_mass": 2.71,
        "left_foot_mass": 1.96,
        "right_hip": .1,
        "right_knee": .1,
        "right_ankle": .1,
        "left_hip": .1,
        "left_knee": .1,
        "left_ankle": .1,
        "ground_r": .5,
        "ground_g": .5,
        "ground_b": .5,
        "body_r": .5,
        "body_g": .5,
        "body_b": .5,
      }
      if dr_option == 'all_dr':
        config.real_dr_list = list(real_dr_values.keys())
      elif dr_option == 'nonconflicting_dr':
        config.real_dr_list = [
          "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "ground_r", "ground_g",
          "ground_b", "body_r", "body_g", "body_b"
        ]
    elif "finger" in config.task:
      real_dr_values = {
        "proximal_mass": .805,
        "distal_mass": .636,
        "spinner_mass": 2.32,
        "proximal_damping": 2.5,
        "distal_damping": 2.5,
        "hinge_damping": .5,
        "ground_r": .5,
        "ground_g": .5,
        "ground_b": .5,
        "finger_r": .5,
        "finger_g": .5,
        "finger_b": .5,
        "hotdog_r": .5,
        "hotdog_g": .5,
        "hotdog_b": .5,
      }
      if dr_option == 'all_dr':
        config.real_dr_list = list(real_dr_values.keys())
      elif dr_option == 'nonconflicting_dr':
        config.real_dr_list = [
          "proximal_mass", "distal_mass", "spinner_mass", "ground_r", "ground_g", "ground_b", "finger_r", "finger_g",
          "finger_b", "hotdog_r", "hotdog_g", "hotdog_b",
        ]
    elif "cheetah" in config.task:
      real_dr_values = {
        "torso_mass": 6.36,
        "bthigh_mass": 1.54,
        "bshin_mass": 1.58,
        "bfoot_mass": 1.07,
        "fthigh_mass": 1.43,
        "fshin_mass": 1.18,
        "ffoot_mass": .85,
        "bthigh_damping": 6,
        "bshin_damping": 4.5,
        "bfoot_damping": 3.,
        "fthigh_damping": 4.5,
        "fshin_damping": 3.,
        "ffoot_damping": 1.5,
        "ground_r": .5,
        "ground_g": .5,
        "ground_b": .5,
        "body_r": .5,
        "body_g": .5,
        "body_b": .5,
      }
    config.real_dr_params = real_dr_values
    config.real_dr_list = list(config.real_dr_params.keys())

    if config.simple_randomization:
      if "cup_catch" in config.task:
        config.real_dr_list = ["ball_mass"]
      elif "walker" in config.task:
        config.real_dr_list = ["torso_mass"]
      elif "finger" in config.task:
        config.real_dr_list = ["distal_mass"]
      elif "cheetah" in config.task:
        config.real_dr_list = ["torso_mass"]
    mean_scale = config.mean_scale
    range_scale = config.range_scale
    config.dr = {}  # (mean, range)
    for key, real_val in config.real_dr_params.items():
      if real_val == 0:
        real_val = 5e-2
      if config.mean_only:
        config.dr[key] = real_val * mean_scale
      else:
        config.dr[key] = (real_val * mean_scale, real_val * range_scale)
    config.sim_params_size = len(config.real_dr_list)

  elif "dummy" in config.task:
    real_dr_values = {
      "square_size": 4,
      "speed_multiplier": 10,
      "square_r": .5,
      "square_g": .5,
      "square_b": 0.0,
    }
    config.real_dr_params = real_dr_values
    config.real_dr_list = list(config.real_dr_params.keys())
    mean_scale = config.mean_scale
    range_scale = config.range_scale
    config.dr = {}  # (mean, range)
    for key, real_val in config.real_dr_params.items():
      if real_val == 0:
        real_val = 5e-2
      if config.mean_only:
        config.dr[key] = real_val * mean_scale
      else:
        config.dr[key] = (real_val * mean_scale, real_val * range_scale)
    config.sim_params_size = len(config.real_dr_list)

  elif config.task in ["gym_FetchPush", "gym_FetchSlide"]:
    config.dr = {
      "body_mass": (1.0, 1.0) # Real parameter is 2.0
    }
  else:
    config.dr = {}
    config.real_dr_list = []

  for k, v in config.dr.items():
    print(k)
    print(v)

  if config.mean_only:
    print("STUFF", config.real_dr_list)
    config.initial_dr_mean = np.array([config.dr[param] for param in config.real_dr_list])
  else:
    config.initial_dr_mean = np.array([config.dr[param][0] for param in config.real_dr_list])
    config.initial_dr_range = np.array([config.dr[param][1] for param in config.real_dr_list])

  return config


def config_debug(config):
  # DEBUG
  config.prefill = 1
  config.steps = 40
  config.deter_size = 2
  config.stoch_size = 3
  config.num_units = 4
  config.cnn_depth = 2
  config.eval_every = 2
  config.log_every = 1
  config.train_every = 3
  config.pretrain = 3
  config.train_steps = 7
  config.time_limit = 15
  config.batch_size = 50
  config.batch_length = 6
  config.update_target_every = 1

  config.num_real_episodes = 20
  config.num_sim_episodes = 20
  # config.range_only = True
  config.num_dr_steps = 1 if config.range_only else 3
  config.starting_mean_scale = 5
  config.starting_range_scale = 1
  config.ending_mean_scale = 1
  config.ending_range_scale = .1
  config.minimal = False

  return config


class Dreamer(tools.Module):

  def __init__(self, config, datadir, actspace, writer, dataset=None, strategy=None):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
    if strategy is None:
      self._strategy = tf.distribute.MirroredStrategy()
    else:
      self._strategy = strategy
    with self._strategy.scope():
      if not config.use_offline_dataset:
        if self._c.outer_loop_version == 2:
          self._train_dataset_sim_only = iter(self._strategy.experimental_distribute_dataset(
              load_dataset(datadir, self._c, use_sim=True, use_real=False)))
        else:
          self._dataset = iter(self._strategy.experimental_distribute_dataset(
            load_dataset(datadir, self._c)))
        self._real_world_dataset = iter(self._strategy.experimental_distribute_dataset(
          load_dataset(datadir, self._c, use_sim=False, use_real=True)))
        self._sim_dataset = iter(self._strategy.experimental_distribute_dataset(
          load_dataset(datadir, self._c, use_sim=True, use_real=False)))
      self._build_model(dataset)

  def __call__(self, obs, reset, dataset=None, state=None, training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    if self._should_train(step):
      log = self._should_log(step)
      n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
      print(f'Training for {n} steps.')
      with self._strategy.scope():
        for train_step in range(n):
          log_images = self._c.log_images and log and train_step == 0
          if self._c.outer_loop_version in [0, 1]:
            self.train(next(self._dataset), log_images)
          else:
            self.train(next(dataset), log_images)
          if (train_step + 1) % self._c.update_target_every == 0:
            self.update_target(self._value, self._target_value)
      if log:
        self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    sys.stdout.flush()
    return action, state

  def train_only(self, dataset):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    step = self._step.numpy().item()  # TODO: not sure if this is being updated
    log = self._should_log(step)
    n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
    print(f'Training for {n} steps.')
    with self._strategy.scope():
      for train_step in range(n):
        log_images = self._c.log_images and log and train_step == 0
        self.train(next(dataset), log_images)
    if log:
      self._write_summaries()
    self._step.assign_add(n)
    sys.stdout.flush()

  @tf.function
  def policy(self, obs, state, training):
    if state is None:
      latent = self._dynamics.initial(obs['image'].shape[0])
      action = tf.zeros((obs['image'].shape[0], self._actdim), self._float)
    else:
      latent, action = state
    embed = self._encode(preprocess(obs, self._c))
    if 'state' in obs:
      state = tf.dtypes.cast(obs['state'], embed.dtype)
      embed = tf.concat([state, embed], axis=-1)
    if 'dr_params' in obs and self._c.outer_loop_version == 2:
      dr_values = obs['dr_params']
      print("DR VALUES", dr_values)
      # If there are no sim params, this is presumably the real world
      if np.prod(dr_values.shape) == 0:
      # if dr_values.size == 0:
        dr_values = tf.expand_dims(self.learned_dr_mean, 0)
      dr_params = tf.dtypes.cast(dr_values, embed.dtype)
      embed = tf.concat([dr_params, embed], axis=-1)
    latent, _ = self._dynamics.obs_step(latent, action, embed)
    feat = self._dynamics.get_feat(latent)
    if training:
      action = self._actor(feat).sample()
    else:
      action = self._actor(feat).mode()
    action = self._exploration(action, training)
    state = (latent, action)
    return action, state

  def load(self, filename):
    super().load(filename)
    self._should_pretrain()

  def update_target(self, original, target):  # TODO: should this be @tf.function?
    target.set_weights(original.get_weights())

  def _random_crop(self, data):
    obs = data['image']
    top_row = tf.repeat(obs[:, :, :1], 4, axis=2)
    bottom_row = tf.repeat(obs[:, :, -1:], 4, axis=2)
    new_img = tf.concat([top_row, obs, bottom_row], axis=2)
    left_row = tf.repeat(new_img[:, :, :, :1], 4, axis=3)
    right_row = tf.repeat(new_img[:, :, :, -1:], 4, axis=3)
    new_img = tf.concat([left_row, new_img, right_row], axis=3)
    b, t, h, w, c = obs.shape
    vert_crop = np.random.randint(0, 9)
    horiz_crop = np.random.randint(0, 9)
    cropped = new_img[:, :, vert_crop:vert_crop + h, horiz_crop: horiz_crop + w]
    data['image'] = cropped

  @tf.function()
  def train(self, data, log_images=False):
    self._strategy.experimental_run_v2(self._train, args=(data, log_images))


  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:

      if self._c.random_crop:
        self._random_crop(data)

      if 'success' in data:
        success_rate = tf.reduce_sum(data['success']) / data['success'].shape[1]
      else:
        success_rate = tf.convert_to_tensor(-1)
      embed = self._encode(data)
      img_embed = tf.identity(embed)
      if 'state' in data:
        embed = tf.concat([data['state'], embed], axis=-1)
      if 'dr_params' in data and self._c.outer_loop_version == 2:
        dr_values = data['dr_params']
        embed = tf.concat([dr_values, embed], axis=-1)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      if self._c.outer_loop_version == 1:
        sim_param_pred = self._sim_params(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      reward_obj = reward_pred.log_prob(data['reward'])

      # Mask out the elements which came from the real world env
      reward_obj = reward_obj * (1 - data['real_world'])

      likes.reward = tf.reduce_mean(reward_obj)
      if self._c.outer_loop_version == 1:
        if self._c.binary_prediction:
          labels = tf.cast(data['sim_params'] > data['distribution_mean'], tf.int32)
          predictions = sim_param_pred.mean()
          sim_param_obj = -tf.keras.losses.binary_crossentropy(labels, predictions)
        else:
          sim_param_obj = sim_param_pred.log_prob(tf.math.log(data['sim_params']))
        sim_param_obj = sim_param_obj * (1 - data['real_world'])
        if self._c.last_param_pred_only:
          sim_param_obj = sim_param_obj[:, -1]
        likes.sim_params = tf.reduce_mean(sim_param_obj)
      if self._c.pcont:
        pcont_pred = self._pcont(feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)

      # weigh losses
      if self._c.outer_loop_version == 1:
        if self._c.individual_loss_scale:
          likes.sim_params *= self._c.sim_params_loss_scale
        else:
          assert self._c.sim_params_loss_scale <= 1

        likes.sim_params *= 1-self._c.sim_params_loss_scale
        likes.reward *= self._c.sim_params_loss_scale
        likes.image *= self._c.sim_params_loss_scale

      model_loss = self._c.kl_scale * div - sum(likes.values())
      model_loss /= float(self._strategy.num_replicas_in_sync)

    if not self._c.use_offline_dataset:
      with tf.GradientTape() as actor_tape:
        imag_feat = self._imagine_ahead(post)
        reward = self._reward(imag_feat).mode()
        if self._c.pcont:
          pcont = self._pcont(imag_feat).mean()
        else:
          pcont = self._c.discount * tf.ones_like(reward)
        value = self._target_value(imag_feat).mode()
        returns = tools.lambda_return(
            reward[:-1], value[:-1], pcont[:-1],
            bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
        discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
            [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
        actor_loss = -tf.reduce_mean(discount * returns)
        actor_loss /= float(self._strategy.num_replicas_in_sync)

      with tf.GradientTape() as value_tape:
        value_pred = self._value(imag_feat)[:-1]
        target = tf.stop_gradient(returns)
        value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
        value_loss /= float(self._strategy.num_replicas_in_sync)

    model_norm = self._model_opt(model_tape, model_loss)
    if not self._c.use_offline_dataset:
      actor_norm = self._actor_opt(actor_tape, actor_loss)
      value_norm = self._value_opt(value_tape, value_loss)
    else:
      # Fill with dummy values so logging isn't broken
      value_norm = 0
      actor_norm = 0
      value_loss = 0
      actor_loss = 0

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm, success_rate)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)

  @tf.function()
  def update_sim_params(self, data, update=True):
    return self._strategy.experimental_run_v2(self._update_sim_params, args=(data, update))

  def _update_sim_params(self, data, update):
    with tf.GradientTape() as sim_param_tape:
      embed = self._encode(data)
      if 'state' in data:
        embed = tf.concat([data['state'], embed], axis=-1)
      dr_mean = tf.exp(self.learned_dr_mean)
      dr_std = tf.maximum(dr_mean * 0.1, 1e-3) #TODO : Change this if needed, corresponds to wrappers.py
      random_num = tf.random.normal(dr_mean.shape, dtype=dr_mean.dtype)
      sampled_dr = random_num * dr_std + dr_mean
      desired_shape = (embed.shape[0], embed.shape[1], dr_mean.shape[0])
      sampled_dr = tf.broadcast_to(sampled_dr, desired_shape)
      embed1 = tf.concat([sampled_dr, embed], axis=-1)
      post, prior = self._dynamics.observe(embed1, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      scale = tf.constant(self._c.sim_param_regularization)
      regularization = scale * tf.norm(dr_mean - config.initial_dr_mean)
      sim_param_loss = -tf.reduce_mean(image_pred.log_prob(data['image'])) + regularization
    if update:
      sim_param_norm = self._dr_opt(sim_param_tape, sim_param_loss, module=False)  # TODO: revert
      self._metrics['sim_param_loss'].update_state(sim_param_loss)
      self._metrics['sim_param_norm'].update_state(sim_param_norm)
      self._metrics['sim_param_loss_regularization'].update_state(regularization)
      for i, key in enumerate(self._c.real_dr_list):
        self._metrics['learned_mean' + key].update_state(dr_mean[i])
    return sim_param_loss


  def _build_model(self, dataset=None):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._dynamics = models.RSSM(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    if self._c.use_depth:
      self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act, shape=(64,64,4))
    else:
      channels = 1 if self._c.grayscale else 3
      self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act, shape=(64, 64, channels))
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    if self._c.outer_loop_version == 1:
      if self._c.binary_prediction:
        self._sim_params = models.DenseDecoder((self._c.sim_params_size,), 2, self._c.num_units, act=act, dist='binary')
      else:
        self._sim_params = models.DenseDecoder((self._c.sim_params_size,), 2, self._c.num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._target_value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    if self._c.outer_loop_version == 1:
      model_modules = [self._encode, self._dynamics, self._decode, self._reward, self._sim_params]
    elif self._c.outer_loop_version in [0, 2]:
      model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    if self._c.outer_loop_version == 2:
      dr_mean = np.array([self._c.dr[k] for k in config.real_dr_list])

      self.learned_dr_mean = tf.Variable(np.log(dr_mean), trainable=True, dtype=tf.float32)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    if self._c.outer_loop_version == 2:
      self._dr_opt = Optimizer('dr', [self.learned_dr_mean], self._c.dr_lr)
      # Do a train step to initialize all variables, including optimizer
      # statistics. Ideally, we would use batch size zero, but that doesn't work
      # in multi-GPU mode.
    if dataset is not None:
      self.train(next(dataset))
    elif self._c.outer_loop_version in [0, 1]:
      self.train(next(self._dataset))
    else:
      self.train(next(self._train_dataset_sim_only))
      self.update_sim_params(next(self._real_world_dataset))
    if not self._c.use_offline_dataset:
      self.update_target(self._value, self._target_value)


  def _exploration(self, action, training):
    if training:
      amount = self._c.expl_amount
      if self._c.expl_decay:
        amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
      if self._c.expl_min:
        amount = tf.maximum(self._c.expl_min, amount)
      self._metrics['expl_amount'].update_state(amount)
    elif self._c.eval_noise:
      amount = self._c.eval_noise
    else:
      return action
    if self._c.expl == 'additive_gaussian':
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    if self._c.expl == 'completely_random':
      return tf.random.uniform(action.shape, -1, 1)
    if self._c.expl == 'epsilon_greedy':
      indices = tfd.Categorical(0 * action).sample()
      return tf.where(
          tf.random.uniform(action.shape[:1], 0, 1) < amount,
          tf.one_hot(indices, action.shape[-1], dtype=self._float),
          action)
    raise NotImplementedError(self._c.expl)

  def _imagine_ahead(self, post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in post.items()}
    policy = lambda state: self._actor(
        tf.stop_gradient(self._dynamics.get_feat(state))).sample()
    states = tools.static_scan(
        lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
        tf.range(self._c.horizon), start)
    imag_feat = self._dynamics.get_feat(states)
    return imag_feat

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, value_loss, actor_loss, model_norm, value_norm,
      actor_norm, success_rate):
    self._metrics['success_rate'].update_state(success_rate)
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())

  def _image_summaries(self, data, embed, image_pred):
    truth = data['image'][:6] + 0.5
    recon = image_pred.mode()[:6]
    init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    init = {k: v[:, -1] for k, v in init.items()}
    prior = self._dynamics.imagine(data['action'][:6, 5:], init)
    openl = self._decode(self._dynamics.get_feat(prior)).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    openl = tf.concat([truth, model, error], 2)
    tools.graph_summary(
        self._writer, tools.video_summary, 'agent/openl', openl)

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items() if v.count > 0]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    sys.stdout.flush()
    self._writer.flush()

  def predict_sim_params(self, obs, reset, state=None):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)

    if state is None:
      latent = self._dynamics.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._actdim), self._float)
    else:
      latent, action = state
    embed = self._encode(preprocess(obs, self._c))
    if 'state' in obs:
      state = tf.dtypes.cast(obs['state'], embed.dtype)
      embed = tf.concat([state, embed], axis=-1)
    latent, _ = self._dynamics.obs_step(latent, action, embed)
    feat = self._dynamics.get_feat(latent)

    action = self._actor(feat).mode()
    action = self._exploration(action, False)
    state = (latent, action)


    sim_param_pred = self._sim_params(feat)
    return  action, state, sim_param_pred


def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
  return obs


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config, use_sim=None, use_real=None):
  episode = next(tools.load_episodes(directory, 1, use_sim=use_sim, use_real=use_real, buffer_size=config.buffer_size))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
    directory, config.train_steps, config.batch_length,
    config.dataset_balance, real_world_prob=config.real_world_prob, use_sim=use_sim, use_real=use_real, buffer_size=config.buffer_size)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset


def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.action_repeat
  ret = episode['reward'].sum()
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'episodes', episodes)]
  if 'success' in episode:
    success = True in episode['success']
    success_str = "succeeded" if success == 1 else "did not succeed"
    metrics.append((f'{prefix}/success', success))
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}, which {success_str}.')
  else:
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  sys.stdout.flush()
  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
    if prefix == 'test' and config.log_images:
      tools.video_summary(f'sim/{prefix}/video', episode['image'][None])


def make_env(config, writer, prefix, datadir, store, index=None, real_world=False):
  suite, task = config.task.split('_', 1)
  if suite == 'peg':
    if config.dr is None or real_world:
      env = wrappers.PegTask(use_state=config.use_state, real_world=real_world)
    else:
      env = wrappers.PegTask(dr=config.dr, use_state=config.use_state, real_world=real_world)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'kitchen':
    if config.dr is None or real_world:
      env = wrappers.Kitchen(use_state=config.use_state, early_termination=config.early_termination, real_world=real_world,
                             dr_shape=config.sim_params_size, dr_list=config.real_dr_list,
                             task=task, simple_randomization=False, step_repeat=config.step_repeat,
                             outer_loop_version=config.outer_loop_version, control_version=config.control_version,
                             step_size=config.step_size, initial_randomization_steps=config.initial_randomization_steps,
                             minimal=config.minimal, grayscale=config.grayscale)
    else:
      env = wrappers.Kitchen(dr=config.dr, mean_only=config.mean_only, anneal_range_scale=config.anneal_range_scale, predict_val=config.predict_val, early_termination=config.early_termination,
                             use_state=config.use_state, real_world=real_world, dr_list=config.real_dr_list,
                             dr_shape=config.sim_params_size, task=task,
                             simple_randomization=config.simple_randomization, step_repeat=config.step_repeat,
                             outer_loop_version=config.outer_loop_version, control_version=config.control_version,
                             step_size=config.step_size, initial_randomization_steps=config.initial_randomization_steps,
                             minimal=config.minimal, grayscale=config.grayscale)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'metaworld':
    if config.dr is None or real_world:
      env = wrappers.MetaWorld(task, use_state=config.use_state, use_img=config.use_img, early_termination=config.early_termination,
                               real_world=real_world, dr_shape=config.sim_params_size, dr_list=config.real_dr_list,
                               simple_randomization=False, outer_loop_version=config.outer_loop_version,
                               use_depth=config.use_depth, grayscale=config.grayscale)
    else:
      env = wrappers.MetaWorld(task, dr=config.dr, mean_only=config.mean_only, early_termination=config.early_termination,
                             use_state=config.use_state,  use_img=config.use_img, real_world=real_world, dr_list=config.real_dr_list,
                             dr_shape=config.sim_params_size, simple_randomization=config.simple_randomization,
                             outer_loop_version=config.outer_loop_version, use_depth=config.use_depth)
      env = wrappers.ActionRepeat(env, config.action_repeat, grayscale=config.grayscale)
      env = wrappers.NormalizeActions(env)
  elif suite == 'dmc':
    if config.dr is None or real_world:
      env = wrappers.DeepMindControl(task, dr=config.dr, use_state=config.use_state, real_world=real_world, dr_shape=config.sim_params_size, dr_list=config.real_dr_list,
                                     simple_randomization=config.simple_randomization, outer_loop_type=config.outer_loop_version, mean_only=config.mean_only)
    else:
      env = wrappers.DeepMindControl(task, dr=config.dr, use_state=config.use_state, dr_shape=config.sim_params_size, dr_list=config.real_dr_list,
                                     real_world=real_world, simple_randomization=config.simple_randomization,
                                     outer_loop_type=config.outer_loop_version, mean_only=config.mean_only)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, (64, 64), grayscale=False,
        life_done=True, sticky_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'gym':
    if index == 0 or index is None: #first index is always real world
      env = wrappers.GymControl(task)
    else:
      env = wrappers.GymControl(task, dr=config.dr)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'dummy':
    if config.dr is None or real_world:
      env = wrappers.Dummy(dr=config.dr, real_world=real_world,
                                     dr_shape=config.sim_params_size, dr_list=config.real_dr_list,
                                     outer_loop_version=config.outer_loop_version, mean_only=config.mean_only)
    else:
      env = wrappers.Dummy(dr=config.dr, dr_shape=config.sim_params_size,
                                     dr_list=config.real_dr_list,
                                     real_world=real_world,
                                     outer_loop_version=config.outer_loop_version, mean_only=config.mean_only)
    env = wrappers.ActionRepeat(env, config.action_repeat)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  callbacks = []
  if store:
    callbacks.append(lambda ep, dataset_step: tools.save_episodes(datadir, [ep], dataset_step))
  callbacks.append(
      lambda ep, _: summarize_episode(ep, config, datadir, writer, prefix))
  env = wrappers.Collect(env, callbacks, config.precision)
  env = wrappers.RewardObs(env)
  return env

def log_memory(step):
  process = psutil.Process(os.getpid())
  memory_use = process.memory_info().rss / float(2 ** 20)
  print("Memory Use MiB", memory_use)
  tf.summary.scalar('general/memory_mib', memory_use, step)


def check_train_with_real(dr_list):
  return False # TODO: remove this line if we want to do this again
  # TODO: this currently won't scale to more dr params. It's just a proof of concept showing we can do this.
  range_ = .02
  duration = 25
  std_cutoff = .01
  timestep_cutoff = 100
  if len(dr_list) < timestep_cutoff:
    return False
  segment = dr_list[-duration:]
  observed_range = np.max(segment) - np.min(segment)
  observed_std = np.std(segment)
  return observed_std <= std_cutoff and observed_range <= range_



def generate_dataset(config, sim_envs, real_envs):
  real_dr_list = config.real_dr_list
  real_dr = config.real_dr_params
  num_real_episodes = config.num_real_episodes
  num_sim_episodes = config.num_sim_episodes
  num_dr_steps = config.num_dr_steps
  episodes_per_dr_step = int(num_sim_episodes / num_dr_steps / config.envs)
  episodes_per_dr_step = max(episodes_per_dr_step, 1)
  starting_mean_scale = config.starting_mean_scale
  starting_range_scale = config.starting_range_scale
  ending_mean_scale = config.ending_mean_scale
  ending_range_scale = config.ending_range_scale
  if num_dr_steps > 1:
    mean_step_size = (ending_mean_scale - starting_mean_scale) / (num_dr_steps - 1)
    range_step_size = (ending_range_scale - starting_range_scale) / (num_dr_steps - 1)
  else:
    mean_step_size = range_step_size = 0

  curr_mean_scale = float(starting_mean_scale)
  curr_range_scale = float(starting_range_scale)

  # Save params
  config.actspace = sim_envs[0].action_space
  with open(config.logdir / "dataset_config.pkl", "wb") as f:
    pkl.dump(config, f)

  action_length = len(sim_envs[0].action_space.sample())
  if action_length == 4:  # Metaworld. probably useless
    bot_agent = lambda o, d, da, s: ([np.array([1, 1, 1, 1]) for _ in d], None)
  elif action_length == 3:  # Kitchen.  Likely only useful for kettle tasks
    bot_agent = lambda o, d, da, s: ([np.array([0, 1, 0]) for _ in d], None)
  for i in range(num_dr_steps):
    dr = {}
    for param in real_dr_list:
      val = real_dr[param]
      if config.mean_only:
        assert not config.range_only
        dr[param] = curr_mean_scale * val
      elif config.range_only:
        if 'kettle_mass' in param:
          dr[param] = (1.15, 0.35)
        elif '_b' in param or '_r' in param or '_g' in param:
          dr[param] = (0.5, 0.17)
        else:
          raise NotImplementedError(f"Can't handle {param}")
      else:
        dr[param] = (val * curr_mean_scale, val * curr_range_scale)
    for env in sim_envs:
      env.set_dr(dr)
      env.apply_dr()
      env.set_dataset_step(i)

    # Update sim params
    for _ in range(episodes_per_dr_step):
      for env in sim_envs:
        env.apply_dr()
      tools.simulate(bot_agent, sim_envs, dataset=None, episodes=config.envs)

    curr_mean_scale += mean_step_size
    curr_range_scale += range_step_size

  # Collect 4 real-world datasets
  if config.range_only:
    dr = {}
    for param in real_dr_list:
      if 'kettle_mass' in param:
        dr[param] = (1.15, 0.35)
      elif '_b' in param or '_r' in param or '_g' in param:
        dr[param] = (0.5, 0.17)
    for env in sim_envs:
      env.reset()
      env.set_dr(dr)
      env.apply_dr()
      env.set_dataset_step("test_med")
      tools.simulate(bot_agent, sim_envs, dataset=None, episodes=num_real_episodes)

    for param in real_dr_list:
      if 'kettle_mass' in param:
        dr[param] = (0.45, 0.35)
      elif '_b' in param or '_r' in param or '_g' in param:
        dr[param] = (0.17, 0.17)
    for env in sim_envs:
      env.reset()
      env.set_dr(dr)
      env.apply_dr()
      env.set_dataset_step("test_low")
      tools.simulate(bot_agent, sim_envs, dataset=None, episodes=num_real_episodes)

    for param in real_dr_list:
      if 'kettle_mass' in param:
        dr[param] = (1.85, 0.35)
      elif '_b' in param or '_r' in param or '_g' in param:
        dr[param] = (0.83, 0.17)
    for env in sim_envs:
      env.reset()
      env.set_dr(dr)
      env.apply_dr()
      env.set_dataset_step("test_high")
      tools.simulate(bot_agent, sim_envs, dataset=None, episodes=num_real_episodes)

    for param in real_dr_list:
      if 'kettle_mass' in param:
        dr[param] = (1.05, 1.05)
      elif '_b' in param or '_r' in param or '_g' in param:
        dr[param] = (0.5, 0.5)
    for env in sim_envs:
      env.reset()
      env.set_dr(dr)
      env.apply_dr()
      env.set_dataset_step("test_all")
      tools.simulate(bot_agent, sim_envs, dataset=None, episodes=num_real_episodes)

  else:
    # Dataset format is (name, mean_multiplier, range_multiplier)
    datasets = [
      ("test_med", 1, config.range_scale),
      ("test_low", (1 - config.mean_scale), config.range_scale),
      ("test_high", (1 + config.mean_scale), config.range_scale),
      ("test_all", 1, 1),
    ]
    for name, mean_scale, range_scale in datasets:
      dr = {}
      for param in real_dr_list:
        real_param = config.real_dr_params[param]
        dr[param] = (real_param * mean_scale, real_param * range_scale)
      for env in sim_envs:
        env.reset()
        env.set_dr(dr)
        env.apply_dr()
        env.set_dataset_step(name)
        tools.simulate(bot_agent, sim_envs, dataset=None, episodes=num_real_episodes)



    real_envs[0].set_dataset_step("actual_test_set")
    tools.simulate(bot_agent, real_envs, dataset=None, episodes=num_real_episodes)

  # Collect validation dataset
  sim_envs[0].set_dataset_step("val")
  tools.simulate(bot_agent, sim_envs, dataset=None, episodes=num_real_episodes)


def train_with_offline_dataset(config, datadir, writer):

  # Check dataset exists
  dataset_datadir = pathlib.Path('.').joinpath('logdir', config.datadir)
  with open(dataset_datadir / "dataset_config.pkl", "rb") as f:
    dataset_config = pkl.load(f)

  writer.flush()
  actspace = dataset_config.actspace

  # Load dataset
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    try:
      test_low_dataset = iter(strategy.experimental_distribute_dataset(
        load_dataset(dataset_datadir / "episodes" / "test_low", config)))
      test_med_dataset = iter(strategy.experimental_distribute_dataset(
        load_dataset(dataset_datadir / "episodes" / "test_med", config)))
      test_high_dataset = iter(strategy.experimental_distribute_dataset(
        load_dataset(dataset_datadir / "episodes" / "test_high", config)))
      test_all_dataset = iter(strategy.experimental_distribute_dataset(
        load_dataset(dataset_datadir / "episodes" / "test_all", config)))
      test_dataset = test_all_dataset
    except Exception as e:
      print("ISSUE LOADING DATASET", e)
      test_dataset = iter(strategy.experimental_distribute_dataset(
        load_dataset(dataset_datadir / "episodes" / "test", config)))
      test_low_dataset = None
      test_med_dataset = None
      test_high_dataset = None
      test_all_dataset = None
    num_train_steps_per_level = int(config.steps / config.train_every / dataset_config.num_dr_steps)
    print("NUM TRAIN STEPS PER LEVEL", num_train_steps_per_level)

  agent = Dreamer(config, datadir, actspace, writer, dataset=test_dataset, strategy=strategy)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  else:
    print("checkpoint not loaded")
    print(config.logdir / 'variables.pkl')
    print((config.logdir / 'variables.pkl').exists())

  # Loop
  for i in range(dataset_config.num_dr_steps):

    train_dataset = iter(strategy.experimental_distribute_dataset(
      load_dataset(dataset_datadir / "episodes" / str(i), config)))
    print("Training with dataset", i)
    for _ in range(num_train_steps_per_level):
      # Train
      agent.train_only(train_dataset)
      step = agent._step.numpy().item()

      if test_low_dataset is not None:
        eval_OL1_offline(agent, train_dataset, [test_low_dataset, test_med_dataset, test_high_dataset, test_all_dataset], writer, step, last_only=config.last_param_pred_only)
      else:
        eval_OL1_offline(agent, train_dataset, test_dataset, writer, step, last_only=config.last_param_pred_only)
      writer.flush()

def generate_videos(train_envs, test_envs, agent, logdir, size=(512, 512), num_rollouts=3):
  # Only use a single env from each set
  train_env = train_envs[-1]
  test_env = test_envs[-1]
  envs = [(train_env, "train_env"), (test_env, "test_env")]
  for env, save_name in envs:
    frames = []
    for _ in range(num_rollouts):
      s = None
      env.apply_dr()
      d = False
      promise = env.reset(blocking=False)
      o = promise()
      i = 0
      while not d:
        o = {k: np.expand_dims(o[k], 0) for k in o}
        # RGB --> BGR
        img = env.render(size=size)[:, :, ::-1]
        frames.append(img)
        a, s = agent.policy(o, s, False)
        a = np.array(a).astype(np.float32)[0]
        o, _, d = env.step(a, blocking=False)()[:3]
        i += 1
      print("Generated run of length", i)


      for _ in range(10):
        frames.append(np.zeros_like(frames[0]))
    fps = 10
    writer = cv2.VideoWriter(os.path.join(logdir, save_name + "_videos.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def eval_OL1_offline(agent, train_dataset, test_datasets, writer, step, last_only):  # kangaroo
  train_distribution = next(train_dataset)['distribution_mean']
  if isinstance(test_datasets, list):
    predict_OL1_offline(agent, test_datasets[0], writer, last_only, "test_low", step, train_distribution)
    predict_OL1_offline(agent, test_datasets[1], writer, last_only, "test_med", step, train_distribution)
    predict_OL1_offline(agent, test_datasets[2], writer, last_only, "test_high", step, train_distribution)
    predict_OL1_offline(agent, test_datasets[3], writer, last_only, "test_all", step, train_distribution)

  else:
    predict_OL1_offline(agent, test_datasets, writer, last_only, "test", step, train_distribution)

  predict_OL1_offline(agent, train_dataset, writer, last_only, "train", step, train_distribution)


def predict_OL1_offline(agent, dataset, writer, last_only, log_prefix, step, train_distribution_mean, data=None):
  if data is None:
    data = next(dataset)
  # range_only = agent._c.range_only

  if agent._c.random_crop:
    agent._random_crop(data)

  embed = agent._encode(data)
  if 'state' in data:
    embed = tf.concat([data['state'], embed], axis=-1)
  post, prior = agent._dynamics.observe(embed, data['action'])
  feat = agent._dynamics.get_feat(post)

  if agent._c.binary_prediction:
    sim_param_pred = tf.cast(tf.round(agent._sim_params(feat).mean()), dtype=tf.int32)
    sim_param_real = tf.cast(data['sim_params'] > train_distribution_mean, tf.int32)
  else:
    sim_param_pred = tf.exp(agent._sim_params(feat).mean())
    sim_param_real = data['sim_params']

  assert np.array_equal(np.min(sim_param_real, axis=1), np.max(sim_param_real, axis=1))
  distribution_mean = train_distribution_mean[:, -1]
  if last_only:
    sim_param_pred = sim_param_pred[:, -1]
    sim_param_real = sim_param_real[:, -1]
  else:
    sim_param_pred = np.mean(sim_param_pred, axis=1)
    sim_param_real = np.mean(sim_param_real, axis=1)
    if agent._c.binary_prediction:
      sim_param_pred = np.round(sim_param_pred)
      sim_param_real = np.round(sim_param_real)

  for i, param in enumerate(agent._c.real_dr_list):
    distribution_mean_i = distribution_mean[:, i]
    pred_mean = sim_param_pred[:, i]
    real_mean = sim_param_real[:, i]
    with writer.as_default():
      tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_pred_mean', np.mean(pred_mean), step)
      tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_real_mean', np.mean(real_mean), step)
      if agent._c.binary_prediction:
        tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_error', np.mean(1 - (pred_mean == real_mean)), step)
      elif not np.mean(distribution_mean_i) == 0:
        tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_error',
                          np.mean((pred_mean - real_mean) / distribution_mean_i), step)

def eval_OL1(agent, eval_envs, train_envs, writer, step, last_only):
  predict_OL1(agent, eval_envs, writer, step, log_prefix="test", last_only=last_only)
  for env in train_envs:
    env.apply_dr()
  predict_OL1(agent, train_envs, writer, step, log_prefix="train", last_only=last_only)

  train_env = train_envs[0]
  for i, param in enumerate(config.real_dr_list):
    if config.mean_only:
      prev_mean = train_env.dr[param]
    else:
      prev_mean, prev_range = train_env.dr[param]

    with writer.as_default():
      tf.summary.scalar(f'agent-sim_param/{param}/train_mean', prev_mean, step)
      if not config.mean_only:
        tf.summary.scalar(f'agent-sim_param/{param}/train_range', prev_range, step)

  for i, param in enumerate(config.real_dr_list):
    real_mean = config.real_dr_params[param]
    with writer.as_default():
      tf.summary.scalar(f'agent-sim_param/{param}/real_mean', real_mean, step)


def predict_OL1(agent, envs, writer, step, log_prefix, last_only, distribution_mean):
  real_pred_sim_params = tools.simulate_real(
    functools.partial(agent, training=False), functools.partial(agent.predict_sim_params), envs, episodes=1,
    last_only=last_only)
  if agent._c.binary_prediction:
    real_pred_sim_params = tf.round(real_pred_sim_params.mean())
  else:
    real_pred_sim_params = tf.exp(real_pred_sim_params)

  real_params = envs[0].get_dr()

  for i, param in enumerate(config.real_dr_list):
    try:
      pred_mean = real_pred_sim_params[i]
    except:
      pred_mean = real_pred_sim_params
    print(f"Learned {param}", pred_mean)

    with writer.as_default():
      tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_pred_mean', pred_mean, step)

      real_dr_param = real_params[i]

      if agent._c.binary_prediction:
        labels = real_dr_param > distribution_mean[i]
        tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_percent_error', np.mean(labels == pred_mean), step)
      else:
        if distribution_mean[i] == 0:
          tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_error',
                            (pred_mean - real_dr_param), step)
        else:
          tf.summary.scalar(f'agent-sim_param/{param}/{log_prefix}_error',
                            (pred_mean - real_dr_param) / distribution_mean[i], step)
    writer.flush()

def main(config):
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  config.steps = int(config.steps)
  config.logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', config.logdir)

  # Create environments.
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()

  train_sim_envs = [wrappers.Async(lambda: make_env(
      config, writer, 'sim_train', datadir, store=True, real_world=False), config.parallel)
      for i in range(config.envs)]
  train_real_envs = [wrappers.Async(lambda: make_env(
    config, writer, 'real_train', datadir, store=True, real_world=True), config.parallel)
                for _ in range(config.envs)]
  test_envs = [wrappers.Async(lambda: make_env(
      config, writer, 'test', datadir, store=False, real_world=True), config.parallel)
      for _ in range(config.envs)]
  actspace = train_sim_envs[0].action_space

  if config.use_offline_dataset:
    train_with_offline_dataset(config, datadir, writer)
    print("Done with sim param learning!")
    return

  if config.generate_dataset:
    generate_dataset(config, train_sim_envs, train_real_envs)
    for env in train_sim_envs + test_envs:
      env.close()
    if train_real_envs is not None:
      for env in train_real_envs:
        env.close()
    print("Done generating dataset!")
    return

  if config.generate_videos:
    print("Collecting a trajectory so it doesn't die on us")
    random_agent = lambda o, d, da, s: ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_sim_envs, None, episodes=1)
    print("Loading past run")
    agent = Dreamer(config, datadir, actspace, writer)
    agent.load(config.logdir / 'variables.pkl')
    print("Generating videos")
    generate_videos(train_sim_envs, test_envs, agent, config.logdir)
    for env in train_sim_envs + test_envs:
      env.close()
    if train_real_envs is not None:
      for env in train_real_envs:
        env.close()
    print("Done generating videos!")
    return

  # Prefill dataset with random episodes.
  step = count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  random_agent = lambda o, d, da, s: ([actspace.sample() for _ in d], None)
  dataset = None
  print(f'Prefill dataset with {prefill} simulated steps.')
  tools.simulate(random_agent, train_sim_envs, dataset, prefill / config.action_repeat)
  num_real_prefill = int(prefill / config.action_repeat / config.sample_real_every)
  if num_real_prefill == 0:
    num_real_prefill += 1
  print(f'Prefill dataset with {num_real_prefill} real world steps.')
  tools.simulate(random_agent, train_real_envs, dataset, episodes=1, steps=num_real_prefill)
  writer.flush()
  train_real_step_target = config.sample_real_every * config.time_limit
  #update_target_step_target = config.update_target_every * config.time_limit

  # Train and regularly evaluate the agent.
  step = count_steps(datadir, config)
  agent = Dreamer(config, datadir, actspace, writer)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  else:
    print("checkpoint not loaded")
    print(config.logdir / 'variables.pkl')
    print((config.logdir / 'variables.pkl').exists())
  print(f'Simulating agent for {config.steps-step} steps.')
  state = None

  if config.outer_loop_version == 2:
    # Initially, don't use the real world samples to train the model since we don't know their sim params.
    # Once the sim params converge, we can change this to True
    dataset = agent._train_dataset_sim_only
  else:
    dataset = None

  while step < config.steps:
    print('Start evaluation.')
    tools.simulate(
        functools.partial(agent, training=False), test_envs, dataset, episodes=1)
    writer.flush()
    train_batch = next(agent._sim_dataset)
    test_batch = next(agent._real_world_dataset)
    last_only = config.last_param_pred_only
    train_distribution = train_batch['distribution_mean']
    predict_OL1_offline(agent, None, writer, last_only, "train", step, train_distribution, data=train_batch)
    predict_OL1_offline(agent, None, writer, last_only, "test", step, train_distribution, data=test_batch)

    steps = config.eval_every // config.action_repeat
    episodes = int(steps / config.time_limit)
    if episodes == 0:
      episodes += 1
    print('Start collection from simulator.')
    for _ in range(episodes):
      # Call apply_dr so each time we collect an episode we sample a different trajectory from the environment
      for env in train_sim_envs:
        env.apply_dr()
      state = tools.simulate(agent, train_sim_envs, dataset, steps, state=state)
    if step >= train_real_step_target and train_real_envs is not None:
      print("Start collection from the real world")
      tools.simulate(agent, train_real_envs, dataset, episodes=config.num_real_world, state=None)
      train_real_step_target += config.sample_real_every * config.time_limit
    step = count_steps(datadir, config)
    agent.save(config.logdir / 'variables.pkl')
    with open(config.logdir / 'dr_dict.pkl', 'wb') as f:
      pkl.dump(config.dr, f)

    # Update target net
    # if step > update_target_step_target:
    #   agent.update_target(agent._value, agent._target_value)
    #   update_target_step_target += config.update_target_every * config.time_limit

    if config.outer_loop_version == 2:
      dataset = agent._train_dataset_sim_only
      tf.summary.scalar('sim/train_with_real', 0, step)

    # Log memory usage
    log_memory(step)

    # after train, update sim params
    if config.outer_loop_version == 2:
      print("UPDATING!")

      for i in range(config.num_dr_grad_steps):
        agent.update_sim_params(next(agent._real_world_dataset))

      for env in train_sim_envs:
        for i, param in enumerate(config.real_dr_list):
          if config.mean_only:
            prev_mean = env.dr[param]
          else:
            prev_mean, prev_range = env.dr[param]
          pred_mean = np.exp(agent.learned_dr_mean.numpy())[i]
          print(f"Learned {param}", pred_mean)
          alpha = config.alpha

          new_mean = prev_mean * (1 - alpha) + alpha * pred_mean
          if config.mean_only:
            env.dr[param] = new_mean
          else:
            env.dr[param] = (new_mean, prev_range)  # TODO: find a better way to handle the case where we only predict mean but we have a range
          # dr_list.append(new_mean)
          with writer.as_default():
            tf.summary.scalar(f'agent-sim_param/{param}/mean', new_mean, step)
            tf.summary.scalar(f'agent-sim_param/{param}/pred_mean', pred_mean, step)

            real_dr_param = config.real_dr_params[param]
            if not real_dr_param == 0:
              tf.summary.scalar(f'agent-sim_param/{param}/percent_error', (new_mean - real_dr_param)/ real_dr_param, step)

            writer.flush()
        env.apply_dr()

    #after train, update sim param
    elif config.outer_loop_version == 1:  # Kangaroo
      train_batch = next(agent._sim_dataset)
      test_batch = next(agent._real_world_dataset)
      last_only = config.last_param_pred_only
      train_distribution = train_batch['distribution_mean']
      predict_OL1_offline(agent, None, writer, last_only, "train", step, train_distribution, data=train_batch)
      predict_OL1_offline(agent, None, writer, last_only, "test", step, train_distribution, data=test_batch)
      real_pred_sim_params = tools.simulate_real(
          functools.partial(agent, training=False), functools.partial(agent.predict_sim_params), test_envs,
        episodes=config.ol1_episodes, last_only=config.last_param_pred_only)
      if config.binary_prediction:
        real_pred_sim_params = tf.round(real_pred_sim_params.mean())
      else:
        real_pred_sim_params = tf.exp(real_pred_sim_params)
      for env in train_sim_envs:
        if env.dr is not None:
          for i, param in enumerate(config.real_dr_list):
            if config.mean_only:
              prev_mean = env.dr[param]
            else:
              prev_mean, prev_range = env.dr[param]
            try:
              pred_mean = real_pred_sim_params[i]
            except:
              pred_mean = real_pred_sim_params
            alpha = config.alpha

            if config.binary_prediction:
              new_mean = prev_mean + alpha * (np.mean(pred_mean) - 0.5) # TODO: tune this
              new_mean = max(new_mean, 1e-3)  #prevent negative means
            else:
              new_mean = prev_mean * (1 - alpha) + alpha * pred_mean
            if config.mean_only:
              env.dr[param] = new_mean
            else:
              env.dr[param] = (new_mean, prev_range)  # TODO: find a better way to handle the case where we only predict mean but we have a range
            with writer.as_default():
              print("NEW MEAN", param, new_mean, step, pred_mean, "!" * 30)
              tf.summary.scalar(f'agent-sim_param/{param}/mean', new_mean, step)
              tf.summary.scalar(f'agent-sim_param/{param}/pred_mean', pred_mean, step)
              if config.anneal_range_scale > 0:
                tf.summary.scalar(f'agent-sim_param/{param}/range', config.anneal_range_scale*(1-float(step/config.steps)), step)

              real_dr_param = config.real_dr_params[param]
              if not np.mean(real_dr_param) == 0:
                tf.summary.scalar(f'agent-sim_param/{param}/sim_param_error',
                                  (new_mean - real_dr_param) /real_dr_param, step)
              else:
                tf.summary.scalar(f'agent-sim_param/{param}/sim_param_error',
                                  (new_mean - real_dr_param), step)

              writer.flush()

          env.cur_step_fraction = float(step/config.steps)
          env.apply_dr()

  for env in train_sim_envs + test_envs:
    env.close()
  if train_real_envs is not None:
    for env in train_real_envs:
      env.close()
  print("All done!")


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  parser.add_argument('--dr', action='store_true', help='If true, test with DR sim environments')
  parser.add_argument('--dr_option', type=str, help='Which DR option to use')
  parser.add_argument('--gpudevice', type=str, default=None, help='cuda visible devices for fair cluster')
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  config = parser.parse_args()
  if "dmc" in config.task:
    print("USING EGL")
    os.environ['MUJOCO_GL'] = 'egl'
  else:
    print("USING OSMESA")
    os.environ['MUJOCO_GL'] = 'osmesa'

  if config.dr:
    config = config_dr(config)
  else:
    config.dr = None
    config.real_dr_list = []

  try:
    print("GPUS found", tf.config.list_physical_devices(device_type="GPU"))
  except:
    print("GPUS found", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

  if config.gpudevice is not None:
    print('Setting gpudevice to:', config.gpudevice)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpudevice
    os.environ['EGL_DEVICE_ID'] = config.gpudevice


  import  wrappers
  suffix = "-dataset" if config.generate_dataset else "-dreamer"
  path_name = config.id + "-" + config.task + suffix
  path = pathlib.Path('.').joinpath('logdir', path_name)
  config.logdir = path
  # Raise an error if this ID is already used, unless we're in debug mode or continuing a previous run
  if 'debug' in config.id:
    config = config_debug(config)
    if path.exists():
      print("Path exists")
      shutil.rmtree(path)
  elif path.exists():
    print("continuing past run", config.id)
    try:
      with open(config.logdir / 'dr_dict.pkl', 'rb') as f:
        dr = pkl.load(f)
        config.dr = dr
    except Exception as e:
      print("Trouble loading dr dictionary", e)
  else:
    print("New run", config.id)
  main(config)
