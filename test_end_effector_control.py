import matplotlib
# matplotlib.use('Qt5Agg')

import numpy as np
import wrappers
import moviepy.editor as mpy
from matplotlib import pyplot as plt



dr_list = [
    "stick_mass",
    "stick_friction",
    "stick_r",
    "stick_g",
    "stick_b",
    "object_mass",
    "object_friction",
    "object_body_r",
    "object_body_g",
    "object_body_b",
    "object_handle_r",
    "object_handle_g",
    "object_handle_b",
    "table_friction",
    "table_r",
    "table_g",
    "table_b",
    "robot_friction",
    "robot_r",
    "robot_g",
    "robot_b",

]


dr_list = [
    "basket_friction",
    "basket_goal_r",
    "basket_goal_g",
    "basket_goal_b",
    "backboard_r",
    "backboard_g",
    "backboard_b",
    "object_mass",
    "object_friction",
    "object_r",
    "object_g",
    "object_b",
    "table_friction",
    "table_r",
    "table_g",
    "table_b",
    "robot_friction",
    "robot_r",
    "robot_g",
    "robot_b",

]

real_dr = {
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
    "table_friction": 2.,
    "table_r": .6,
    "table_g": .6,
    "table_b": .5,
    "robot_friction": 1.,
    "robot_r": .5,
    "robot_g": .1,
    "robot_b": .1,
}

real_dr = {
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
    "table_friction": 2.,
    "table_r": .6,
    "table_g": .6,
    "table_b": .5,
    "robot_friction": 1.,
    "robot_r": .5,
    "robot_g": .1,
    "robot_b": .1,
}


dr = {}
for k, v in real_dr.items():
    dr[k] = (v * 2 + 200, .0001)

env1 = wrappers.MetaWorld(name="basketball", size=(512, 512), dr_list=dr_list, dr=dr)
o = env1.render(mode='rgb_array')
plt.imshow(o)
plt.show()
model = env1._env.sim.model
x = 3



#
real_dr_params = {
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
        "kettle_mass": 0.02,
        "kettle_r": 0.5,
        "knob_mass": 0.02,
        "lighting": 0.3,

    "microwave_b": 0.5,
    "microwave_friction": 1,
    "microwave_g": 0.5,
    "microwave_mass": .26,
    "microwave_r": 0.5,
        "robot_b": 0.95,
        "robot_friction": 1.0,
        "robot_g": 0.95,
        "robot_r": 0.95,
        "stove_b": 0.5,
        "stove_friction": 1.,
        "stove_g": 0.5,
        "stove_r": 0.5,

      }
dr = {}
offset = 1.5
# offset = .98
range1 = 1
for key, real_val in real_dr_params.items():
    dr[key] = (real_val * offset + (real_val * range1), 0)

env = wrappers.Kitchen(size=(512, 512), step_repeat=50, control_version='mocap_ik', dr=dr, task='kitchen_open_cabinet')
# env1.reset()
env.reset()
geom_dict = env._env.sim.model._geom_name2id

microwave_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_viz" in name]
orig_rgb = env._env.sim.model.geom_rgba[microwave_viz_indices]
# env._env.sim.model.geom_rgba[microwave_viz_indices, 1] = env._env.sim.model.geom_rgba[microwave_viz_indices, 1] * 10


microwave_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_collision" in name]
microwave_index = env._env.sim.model.body_name2id('slidelink')
d = env._env.sim.model.body_name2id

orig_mass = env._env.sim.model.body_mass[microwave_index: microwave_index + 1]
orig_friction = env._env.sim.model.geom_friction[microwave_collision_indices, 0]


# env._env.sim.model.body_mass[microwave_index: microwave_index + 1] = env._env.sim.model.body_mass[microwave_index: microwave_index + 1] * 10
# env._env.sim.model.geom_friction[microwave_collision_indices, 0] = env._env.sim.model.geom_friction[microwave_collision_indices, 0] * 10


# microwave_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_viz" in name]
# orig_rgb = env._env.sim.model.geom_rgba[microwave_viz_indices]
# env._env.sim.model.geom_rgba[microwave_viz_indices, 1] = env._env.sim.model.geom_rgba[microwave_viz_indices, 1] * 10
#
#
# microwave_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_collision" in name]
# microwave_index = env._env.sim.model.body_name2id('microdoorroot')
# d = env._env.sim.model.body_name2id
#
# orig_mass = env._env.sim.model.body_mass[microwave_index: microwave_index + 1]
# orig_friction = env._env.sim.model.geom_friction[microwave_collision_indices, 0]
#
#
# env._env.sim.model.body_mass[microwave_index: microwave_index + 1] = env._env.sim.model.body_mass[microwave_index: microwave_index + 1] * 10
# env._env.sim.model.geom_friction[microwave_collision_indices, 0] = env._env.sim.model.geom_friction[microwave_collision_indices, 0] * 10

# img1 = env1.render().copy()
img2 = env.render().copy()
plt.imshow(img2)
# plt.imshow(np.concatenate([img1, img2], axis=1))
plt.show()
x = 3
# #
# # env = env2
#
# # import copy
# #
# # env2 = copy.deepcopy(env1)
# # env2._env.sim.model.geom_rgba[2:33:2, 1] = env2._env.sim.model.geom_rgba[2:33:2, 1] * 2
# # img1 = env1.render()
# # img2 = env2.render()
# # plt.imshow(np.concatenate([img1, img2], axis=1))
# # plt.show()
#
#
# # env2._env.sim.model.dof_damping[:] = env2._env.sim.model.dof_damping[:] * 100
# # env2._env.sim.model.actuator_gainprm[:, 0:1] = env2._env.sim.model.actuator_gainprm[:, 0:1] * 100
# # for _ in range(3):
# #     env1.step(np.array([1, 0, 0]))
# #     env2.step(np.array([1, 0, 0]))
# # end_effector_index = env1.end_effector_index
# # ee1 = env1._env.sim.data.site_xpos[end_effector_index].copy()
# # ee2 = env2._env.sim.data.site_xpos[end_effector_index].copy()
# # diff = ee1 - ee2
#
# def euler2quat(euler):
#     """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
#     euler = np.asarray(euler, dtype=np.float64)
#     assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)
#
#     ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
#     si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
#     ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
#     cc, cs = ci * ck, ci * sk
#     sc, ss = si * ck, si * sk
#
#     quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
#     quat[..., 0] = cj * cc + sj * ss
#     quat[..., 3] = cj * sc - sj * cs
#     quat[..., 2] = -(cj * ss + sj * cc)
#     quat[..., 1] = cj * cs - sj * sc
#     return quat
#
#
env = wrappers.Kitchen(size=(512, 512),control_version='mocap_ik', task='open_cabinet', simple_randomization=True)
# env._env.render(mode='human')
env.reset()  # [-0.108  0.607  2.6  ]
# mocap_index = env.mocap_index
# end_effector_index = env.end_effector_index
# mocap_pos = env._env.sim.data.site_xpos[mocap_index].copy()
# mocap_quat = env._env.sim.data.mocap_quat[0]
while True:

    plt.imshow(env.render())
    plt.show()
    env.step(env.action_space.sample())


    mocap_pos = env._env.sim.data.site_xpos[mocap_index].copy()
    mocap_quat = env._env.sim.data.mocap_quat[0]
    print("MOCAP POS", mocap_pos)
    print("MOCAP QUAT", mocap_quat)
    # print("MOCAP", env._env.data.mocap_pos)
    print("Endeff pos", env._env.sim.data.site_xpos[end_effector_index].copy())  # CORRECT ONE
    # print("Endeff pos2", env._env.sim.model.site_pos[end_effector_index].copy())
    # print("Cabinet pos", env._env.sim.data.site_xpos[env._env.sim.model._site_name2id['cabinet_door']])
    # print("Cabinet goal", env._env.sim.data.site_xpos[env._env.sim.model._site_name2id['cabinet_door_goal']])
    print("Microwave pos", env._env.sim.data.site_xpos[env._env.sim.model._site_name2id['microwave_door']])
    env.get_reward()


    var = input('var')
    val = float(input('val'))
    vars = ['x', 'y', 'z', 'g']
    vars = ['a', 'b', 'c', 'd']
    a = float(input('a'))
    b = float(input('b'))
    c = float(input('c'))
    d = float(input('d'))

    if var == 'g':
        env.set_gripper(val)
    else:
        var_index = vars.index(var)
        mocap_pos[var_index] = val
        env._env.data.set_mocap_pos('mocap', mocap_pos)
    env._env.data.set_mocap_quat('mocap', np.array([0.93937271, 0., 0., -0.34289781]))
    print("SET MOCAP", env._env.sim.data.site_xpos[mocap_index].copy())

    euler_angle = np.array([a, b, c])
    quat_angle = euler2quat(euler_angle)


    var_index = vars.index(var)
    mocap_quat[var_index] = val
    env._env.data.set_mocap_quat('mocap', quat_angle)
    env._env.data.set_mocap_quat('mocap', np.array([.653, -.271, .653, -.271]))
    env._env.data.set_mocap_quat('mocap', np.array([a, b, c, d]))

    for _ in range(2000):
        # print("sim step", env._env.sim.data.site_xpos[mocap_index].copy(), env._env.sim.data.site_xpos[end_effector_index].copy())
        try:
            env._env.sim.step()
        except:
            print("????")
#

env = wrappers.Kitchen(size=(512, 512), step_repeat=50, control_version='mocap_ik')
env._env.sim.model.body_mass[48:49] = 99
# env._env.sim.model.body_mass[49:50] = 99

# ====================== check whether end_effector control works ================
# Axis aligned
img_size = 512
end_effector_index = env.end_effector_index
for i in range(2, 3):
    print("TEST", i)
    o = env.reset()
    offset = np.zeros((3,))
    offset[i] = 1  # Will be scaled down by the step size
    frames = [env.render(size=(img_size,img_size))]
    positions = [env._env.sim.data.site_xpos[end_effector_index].copy()]
    a = np.zeros((3,))
    a[1] = 1
    a[0] = .5
    # a[2] = 1
    # a[i] = 1  # Specify a change along one axis. We could also comment this out to check that with no change the arm stays still.
    for k in range(35):
        o, _, _, _ = env.step(a)
        frames.append(env.render(size=(img_size,img_size)).copy())
        positions.append(env._env.sim.data.site_xpos[end_effector_index].copy())
    fps = 5
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_gif('test_end_effector_axis_orig' + str(i) + '.gif', fps=fps)
    # We'd expect to see a graph where all positions stay the same except the one along which we're moving.
    for j in range(3):
        data = [p[j] for p in positions]
        plt.plot(range(len(data)), data)
    plt.legend(['x', 'y', 'z'])
    plt.savefig('plot_end_effector_axis' + str(i) + '.png')
    plt.show()
    plt.close()
