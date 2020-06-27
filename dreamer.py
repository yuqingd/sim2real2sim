import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import shutil
import psutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers


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
  config.time_limit = 1000
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
  config.use_state = False
  config.num_dr_grad_steps = 100
  config.control_version = 'metaworld_ik'
  config.generate_videos = False  # If true, it doesn't train; just generates videos

  # Sim2real transfer
  config.real_world_prob = -1   # fraction of samples trained on which are from the real world (probably involves oversampling real-world samples)
  config.sample_real_every = 2  # How often we should sample from the real world
  config.simple_randomization = False

  # these values are for testing dmc_cup_catch
  config.mass_mean = 0.2
  config.mass_range = 0.01

  config.outer_loop_version = 0  # 0= no outer loop, 1 = regression, 2 = conditioning
  config.alpha = 0.5
  config.sim_params_size = 0

  return config

def config_dr(config):
  dr_option = config.dr_option
  if 'kitchen' in config.task:
    if config.simple_randomization:
      config.real_dr_params = {
        "kettle_mass": .88
      }
      config.dr = {  # (mean, range)
        "kettle_mass": (config.mass_mean, config.mass_range)
      }
      config.sim_params_size = 2
    else:
      config.real_dr_params = {
        # "joint1_actuation": 30,
        # "joint2_actuation": 20,
        # "joint3_actuation": 15,
        # "joint4_actuation": 10,
        # "joint5_actuation": 5,
        # "joint6_actuation": 2,
        # "joint7_actuation": 5,
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
        "kettle_mass": 0.02,
        "kettle_r": 0.5,
        "knob_mass": 0.02,
        "lighting": 0.3,
        "robot_b": 0.95,
        "robot_friction": 1.0,
        "robot_g": 0.95,
        "robot_r": 0.95,
        "stove_b": 0.5,
        "stove_friction": 1.,
        "stove_g": 0.5,
        "stove_r": 0.5,

      }
      config.sim_params_size = 2 * len(config.real_dr_params.keys())
      if dr_option == 'accurate_small_range':
        range_scale = 0.1
        config.dr = {}  # (mean, range)
        for key, real_val in config.real_dr_params.items():
          config.dr[key] = (real_val, real_val * range_scale)
      if dr_option == 'inaccurate_small_range':
        range_scale = 0.1
        offset = 1.08
        config.dr = {}  # (mean, range)
        for key, real_val in config.real_dr_params.items():
          config.dr[key] = (real_val * offset, real_val * range_scale)
      else:
        raise NotImplementedError

  elif config.task == 'metaworld_reach':
      return {}
  elif config.task == "dmc_cup_catch":
    print(type(config.simple_randomization))
    if config.simple_randomization:
      config.real_dr_params = {
        "ball_mass": .065
      }
      config.dr = {  # (mean, range)
        "ball_mass": (config.mass_mean, config.mass_range)  # Real parameter is .065
      }
      config.sim_params_size = 2
    else:
      real_ball_mass = .065
      real_actuator_gain = 1
      real_damping = 3
      real_friction = 1
      # real_string_length = .292
      # real_string_stiffness = 0
      # real_ball_size = .025
      config.real_dr_params = {
        "actuator_gain": real_actuator_gain,
        "ball_mass": real_ball_mass,
        # "ball_size": real_ball_size,
        "damping": real_damping,
        "friction": real_friction,
        # "string_length": real_string_length,
        # "string_stiffness": real_string_stiffness,
      }
      config.sim_params_size = 2 * 4
      if dr_option == 'accurate_small_range':
        range_scale = 0.05
        config.dr = {  # (mean, range)
          "actuator_gain": (real_actuator_gain, real_actuator_gain * range_scale),
          "ball_mass": (real_ball_mass, real_ball_mass * range_scale),
          # "ball_size": (real_ball_size, real_ball_size * range_scale),
          "damping": (real_damping, real_damping * range_scale),
          "friction": (real_friction, real_friction * range_scale),
          # "string_length": (real_string_length, real_string_length * range_scale),
          # "string_stiffness": (1e-6, 0.001),
        }
      elif dr_option == 'accurate_large_range':
        range_scale = 5
        config.dr = {  # (mean, range)
          "actuator_gain": (real_actuator_gain, real_actuator_gain * range_scale),
          "ball_mass": (real_ball_mass, real_ball_mass * range_scale),
          # "ball_size": (real_ball_size, real_ball_size * range_scale),
          "damping": (real_damping, real_damping * range_scale),
          "friction": (real_friction, real_friction * range_scale),
          # "string_length": (real_string_length, real_string_length * range_scale),
          # "string_stiffness": (1e-6, .1),
        }
      elif dr_option == 'inaccurate_easy_small_range':
        range_scale = .05
        scale_factor_low = 0.5
        scale_factor_high = 1 / scale_factor_low
        config.dr = {  # (mean, range)
          "actuator_gain": (real_actuator_gain * scale_factor_high, real_actuator_gain * range_scale),
          "ball_mass": (real_ball_mass * scale_factor_low, real_ball_mass * range_scale),
          # "ball_size": (real_ball_size * scale_factor_high, real_ball_size * range_scale),
          "damping": (real_damping * scale_factor_low, real_damping * range_scale),
          "friction": (real_friction * scale_factor_high, real_friction * range_scale),
          # "string_length": (real_string_length * scale_factor_low, real_string_length * range_scale),
          # "string_stiffness": (real_string_stiffness + .01, .1),
        }
      elif dr_option == 'inaccurate_easy_large_covering_range':
        range_scale = 2
        scale_factor_low = 0.5
        scale_factor_high = 1 / scale_factor_low
        config.dr = {  # (mean, range)
          "actuator_gain": (real_actuator_gain * scale_factor_high, real_actuator_gain * range_scale),
          "ball_mass": (real_ball_mass * scale_factor_low, real_ball_mass * range_scale),
          # "ball_size": (real_ball_size * scale_factor_high, real_ball_size * range_scale),
          "damping": (real_damping * scale_factor_low, real_damping * range_scale),
          "friction": (real_friction * scale_factor_high, real_friction * range_scale),
          # "string_length": (real_string_length * scale_factor_low, real_string_length * range_scale),
          # "string_stiffness": (real_string_stiffness + .01, .1),
        }
      elif dr_option == 'inaccurate_easy_large_noncovering_range':
        raise NotImplementedError
      elif dr_option == 'inaccurate_hard_small_range':
        raise NotImplementedError
      elif dr_option == 'inaccurate_hard_large_covering_range':
        raise NotImplementedError
      elif dr_option == 'inaccurate_hard_large_noncovering_range':
        raise NotImplementedError
      else:
        raise ValueError("invalid dr option " + str(dr_option))
  elif config.task in ["gym_FetchPush", "gym_FetchSlide"]:
    config.dr = {
      "body_mass": (1.0, 1.0) # Real parameter is 2.0
    }
  else:
    config.dr = {}

  for k, v in config.dr.items():
    print(k)
    print(np.log(v[0]), np.log(v[1]))

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

  return config


class Dreamer(tools.Module):

  def __init__(self, config, datadir, actspace, writer):
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
    self._strategy = tf.distribute.MirroredStrategy()
    with self._strategy.scope():
      if self._c.outer_loop_version == 2:
        self._train_dataset_sim_only = iter(self._strategy.experimental_distribute_dataset(
            load_dataset(datadir, self._c, use_sim=True, use_real=False)))
        # self._train_dataset_combined = iter(self._strategy.experimental_distribute_dataset(
        #   load_dataset(datadir, self._c, use_sim=True, use_real=True)))
        self._real_world_dataset = iter(self._strategy.experimental_distribute_dataset(
            load_dataset(datadir, self._c, use_sim=False, use_real=True)))
      else:
        self._dataset = iter(self._strategy.experimental_distribute_dataset(
          load_dataset(datadir, self._c)))
      self._build_model()

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
      if log:
        self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    sys.stdout.flush()
    return action, state

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

  @tf.function()
  def train(self, data, log_images=False):
    self._strategy.experimental_run_v2(self._train, args=(data, log_images))

  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:
      if 'success' in data:
        success_rate = tf.reduce_sum(data['success']) / data['success'].shape[1]
      else:
        success_rate = tf.convert_to_tensor(-1)
      embed = self._encode(data)
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
        sim_param_obj = sim_param_pred.log_prob(data['sim_params'])
        sim_param_obj = sim_param_obj * (1 - data['real_world'])

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
      model_loss = self._c.kl_scale * div - sum(likes.values())
      model_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as actor_tape:
      imag_feat = self._imagine_ahead(post)
      reward = self._reward(imag_feat).mode()
      if self._c.pcont:
        pcont = self._pcont(imag_feat).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      value = self._value(imag_feat).mode()
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
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)

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
      dr_std = tf.exp(self.learned_dr_std)
      random_num = tf.random.normal(dr_mean.shape, dtype=dr_mean.dtype)
      sampled_dr = random_num * dr_std + dr_mean
      desired_shape = (embed.shape[0], embed.shape[1], dr_mean.shape[0])
      sampled_dr = tf.broadcast_to(sampled_dr, desired_shape)
      embed = tf.concat([sampled_dr, embed], axis=-1)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      sim_param_loss = -tf.reduce_mean(image_pred.log_prob(data['image']))
    if update:
      sim_param_norm = self._dr_opt(sim_param_tape, sim_param_loss, module=False)
      self._metrics['sim_param_loss'].update_state(sim_param_loss)
      self._metrics['sim_param_norm'].update_state(sim_param_norm)
      for i, key in enumerate(self._c.dr.keys()):
        self._metrics['learned_ + ' + key].update_state(dr_mean[i])
        self._metrics['learned_std' + key].update_state(dr_std[i])
    return sim_param_loss


  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._dynamics = models.RSSM(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    if self._c.outer_loop_version == 1:
      self._sim_params = models.DenseDecoder((self._c.sim_params_size,), 2, self._c.num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
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
      dr_mean = np.array([self._c.dr[k][0] for k in sorted(self._c.dr.keys())])
      dr_range = np.array([self._c.dr[k][0] for k in sorted(self._c.dr.keys())])

      self.learned_dr_mean = tf.Variable(np.log(dr_mean), trainable=True, dtype=tf.float32)
      self.learned_dr_std = tf.Variable(np.log(dr_range), trainable=True, dtype=tf.float32)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    if self._c.outer_loop_version == 2:
      self._dr_opt = Optimizer('dr', [self.learned_dr_mean, self.learned_dr_std], self._c.dr_lr)
      # Do a train step to initialize all variables, including optimizer
      # statistics. Ideally, we would use batch size zero, but that doesn't work
      # in multi-GPU mode.
    if self._c.outer_loop_version in [0, 1]:
      self.train(next(self._dataset))
    else:
      self.train(next(self._train_dataset_sim_only))
      # self.train(next(self._train_dataset_combined))
      self.update_sim_params(next(self._real_world_dataset))


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
  if config.outer_loop_version == 1:
    episode = next(tools.load_episodes(directory, 1))
  else:
    episode = next(tools.load_episodes(directory, 1, use_sim=use_sim, use_real=use_real))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  if config.outer_loop_version == 1:
    generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance, real_world_prob=config.real_world_prob)
  else:
    generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance, real_world_prob=config.real_world_prob, use_sim=use_sim, use_real=use_real)
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
    if prefix == 'test':
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
      env = wrappers.Kitchen(use_state=config.use_state, real_world=real_world, dr_shape=config.sim_params_size,
                             task=task, simple_randomization=config.simple_randomization,
                             outer_loop_version=config.outer_loop_version, control_version=config.control_version)
    else:
      env = wrappers.Kitchen(dr=config.dr, use_state=config.use_state, real_world=real_world,
                             dr_shape=config.sim_params_size, task=task,
                             simple_randomization=config.simple_randomization,
                             outer_loop_version=config.outer_loop_version, control_version=config.control_version)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'metaworld':
    if config.dr is None or real_world:
      env = wrappers.MetaWorld(task, use_state=config.use_state, real_world=real_world)
    else:
      env = wrappers.MetaWorld(task, dr=config.dr, use_state=config.use_state,
                                     real_world=real_world)
  elif suite == 'dmc':
    if config.dr is None or real_world:
      env = wrappers.DeepMindControl(task, use_state=config.use_state, real_world=real_world, dr_shape=config.sim_params_size,
                                     simple_randomization=config.simple_randomization, outer_loop_type=config.outer_loop_version)
    else:
      env = wrappers.DeepMindControl(task, dr=config.dr, use_state=config.use_state, dr_shape=config.sim_params_size,
                                     real_world=real_world, simple_randomization=config.simple_randomization, outer_loop_type=config.outer_loop_version)
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
    if config.dr is None or real_world: #first index is always real world
      env = wrappers.Dummy(task, use_state=config.use_state, real_world=real_world)
    else:
      env = wrappers.Dummy(task, dr=config.dr, use_state=config.use_state,
                                     real_world=real_world)
    env = wrappers.ActionRepeat(env, config.action_repeat)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(
      lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
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


def generate_videos(train_envs, test_envs, agent, size=(512, 512), num_rollouts=3):
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
      while not d:
        o = {k: np.expand_dims(o[k], 0) for k in o}
        frames.append(env.render(size=size))
        a, s = agent.policy(o, s, False)
        a = np.array(a).astype(np.float32)[0]
        o, _, d = env.step(a, blocking=False)()[:3]


      for _ in range(10):
        frames.append(np.zeros_like(frames[0]))
    frames = np.stack(frames)
    frames = np.expand_dims(frames, 0)
    tools.video_summary(f'sim/{save_name}/video', frames)


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
  if config.real_world_prob > 0 or config.outer_loop_version in [1, 2]:
    train_real_envs = [wrappers.Async(lambda: make_env(
      config, writer, 'real_train', datadir, store=True, real_world=True), config.parallel)
                  for _ in range(config.envs)]
  else:
    train_real_envs = None
  test_envs = [wrappers.Async(lambda: make_env(
      config, writer, 'test', datadir, store=False, real_world=True), config.parallel)
      for _ in range(config.envs)]
  actspace = train_sim_envs[0].action_space

  if config.generate_videos:
    random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_sim_envs, None, episodes=1)
    agent = Dreamer(config, datadir, actspace, writer)
    agent.load(config.logdir / 'variables.pkl')
    generate_videos(train_sim_envs, test_envs, agent)
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
  random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
  dataset = None
  print(f'Prefill dataset with {prefill} simulated steps.')
  tools.simulate(random_agent, train_sim_envs, dataset, prefill / config.action_repeat)
  if config.outer_loop_version == 2:
    if train_real_envs is not None:
      num_real_prefill = int(prefill / config.action_repeat / config.sample_real_every)
      if num_real_prefill == 0:
        num_real_prefill += 1
      print(f'Prefill dataset with {num_real_prefill} real world steps.')
      tools.simulate(random_agent, train_real_envs, dataset, num_real_prefill)
  writer.flush()
  train_real_step_target = config.sample_real_every * config.time_limit

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
    dr_list = []
  else:
    dataset = None

  while step < config.steps:
    print('Start evaluation.')
    tools.simulate(
        functools.partial(agent, training=False), test_envs, dataset, episodes=1)
    writer.flush()
    steps = config.eval_every // config.action_repeat
    print('Start collection from simulator.')
    state = tools.simulate(agent, train_sim_envs, dataset, steps, state=state)
    if step >= train_real_step_target and train_real_envs is not None:
      print("Start collection from the real world")
      state = tools.simulate(agent, train_real_envs, dataset, episodes=1, state=state)
      train_real_step_target += config.sample_real_every * config.time_limit
    step = count_steps(datadir, config)
    agent.save(config.logdir / 'variables.pkl')

    if config.outer_loop_version == 2:
      # train_with_real = check_train_with_real(dr_list)
      # if train_with_real:
      #   dataset = agent._train_dataset_combined
      #   tf.summary.scalar('sim/train_with_real', 1, step)
      # else:
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
        for i, param in enumerate(sorted(config.dr.keys())):

          prev_mean, prev_range = env.dr[param]
          pred_mean = np.exp(agent.learned_dr_mean.numpy())[i]
          pred_range = np.exp(agent.learned_dr_std.numpy())[i]
          print(f"Learned {param}", pred_mean, pred_range)
          alpha = config.alpha

          new_mean = prev_mean * (1 - alpha) + alpha * pred_mean
          new_range = prev_range * (1 - alpha) + alpha * pred_range
          env.dr[param] = (new_mean, new_range)
          # dr_list.append(new_mean)
          with writer.as_default():
            tf.summary.scalar(f'agent-sim_param/{param}/mean', new_mean, step)
            tf.summary.scalar(f'agent-sim_param/{param}/range', new_range, step)

            real_dr_param = config.real_dr_params[param]
            if not real_dr_param == 0:
              tf.summary.scalar(f'agent-sim_param/{param}/percent_error', (new_mean - real_dr_param)/ real_dr_param, step)

            writer.flush()
        env.apply_dr()

    #after train, update sim params
    elif config.outer_loop_version == 1:
      real_pred_sim_params = tools.simulate_real(
          functools.partial(agent, training=False), functools.partial(agent.predict_sim_params), test_envs, episodes=1)
      for env in train_sim_envs:
        if env.dr is not None:
          for i, param in enumerate(sorted(config.dr.keys())):
            prev_mean, prev_range = env.dr[param]
            pred_mean = real_pred_sim_params[i * 2]
            pred_range = real_pred_sim_params[i * 2 + 1]
            print(f"Learned {param}", pred_mean, pred_range)
            alpha = config.alpha

            new_mean = prev_mean * (1 - alpha) + alpha * pred_mean
            new_range = prev_range * (1 - alpha) + alpha * pred_range
            env.dr[param] = (new_mean, new_range)
            with writer.as_default():
              tf.summary.scalar(f'agent/sim_param/{param}/mean', new_mean, step)
              tf.summary.scalar(f'agent/sim_param/{param}/range', new_range, step)

              real_dr_param = config.real_dr_params[param]
              if not real_dr_param == 0:
                tf.summary.scalar(f'agent-sim_param/{param}/percent_error', (new_mean - real_dr_param) / real_dr_param,
                                  step)
              writer.flush()

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
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  config = parser.parse_args()

  if config.dr:
    config = config_dr(config)
  else:
    config.dr = {}

  print("GPUS found", tf.config.list_physical_devices(device_type="GPU"))

  path = pathlib.Path('.').joinpath('logdir', config.id + "-" + config.task + "-dreamer")
  # Raise an error if this ID is already used, unless we're in debug mode or continuing a previous run
  if 'debug' in config.id:
    config = config_debug(config)
    if path.exists():
      print("Path exists")
      shutil.rmtree(path)
  elif path.exists():
    print("continuing past run", config.id)
  else:
    print("New run", config.id)
  config.logdir = path
  main(config)
