import os
import numpy as np

from gym import utils
from environments import fetch_env
from gym.envs import robotics

# Ensure we get the path separator correct on windows
FETCH_PATH = os.path.dirname(robotics.__file__)
MODEL_XML_PATH = os.path.join(FETCH_PATH, 'assets/fetch/slide.xml')

class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, use_vision=True, reward_type='dense', deterministic=False, distance_threshold=0.1):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }

        fetch_env.FetchEnv.__init__(
                self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
                gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
                obj_range=0.1, target_range=0.3, distance_threshold=distance_threshold,
                initial_qpos=initial_qpos, reward_type=reward_type, use_vision=use_vision, deterministic=deterministic)
        utils.EzPickle.__init__(self, use_vision, reward_type, deterministic, distance_threshold)