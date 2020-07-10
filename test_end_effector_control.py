import matplotlib
# matplotlib.use('Qt5Agg')

import numpy as np
import wrappers
import moviepy.editor as mpy
from matplotlib import pyplot as plt


# env1 = wrappers.Kitchen(size=(512, 512), step_repeat=50, control_version='mocap_ik')




#
# real_dr_params = {
#         "joint1_damping": 10,
#         "joint2_damping": 10,
#         "joint3_damping": 5,
#         "joint4_damping": 5,
#         "joint5_damping": 5,
#         "joint6_damping": 2,
#         "joint7_damping": 2,
#         "kettle_b": 0.5,
#         "kettle_friction": 1.0,
#         "kettle_g": 0.5,
#         "kettle_mass": 0.02,
#         "kettle_r": 0.5,
#         "knob_mass": 0.02,
#         "lighting": 0.3,
#         "robot_b": 0.95,
#         "robot_friction": 1.0,
#         "robot_g": 0.95,
#         "robot_r": 0.95,
#         "stove_b": 0.5,
#         "stove_friction": 1.,
#         "stove_g": 0.5,
#         "stove_r": 0.5,
#
#       }
# dr = {}
# offset = 1.5
# # offset = .98
# range1 = 1
# for key, real_val in real_dr_params.items():
#     dr[key] = (real_val * offset + (real_val * range1), 0)
#
# env2 = wrappers.Kitchen(size=(512, 512), step_repeat=50, control_version='mocap_ik', dr=dr)
# env1.reset()
# env2.reset()
# img1 = env1.render().copy()
# img2 = env2.render().copy()
# plt.imshow(np.concatenate([img1, img2], axis=1))
# plt.show()
#
# env = env2

# import copy
#
# env2 = copy.deepcopy(env1)
# env2._env.sim.model.geom_rgba[2:33:2, 1] = env2._env.sim.model.geom_rgba[2:33:2, 1] * 2
# img1 = env1.render()
# img2 = env2.render()
# plt.imshow(np.concatenate([img1, img2], axis=1))
# plt.show()


# env2._env.sim.model.dof_damping[:] = env2._env.sim.model.dof_damping[:] * 100
# env2._env.sim.model.actuator_gainprm[:, 0:1] = env2._env.sim.model.actuator_gainprm[:, 0:1] * 100
# for _ in range(3):
#     env1.step(np.array([1, 0, 0]))
#     env2.step(np.array([1, 0, 0]))
# end_effector_index = env1.end_effector_index
# ee1 = env1._env.sim.data.site_xpos[end_effector_index].copy()
# ee2 = env2._env.sim.data.site_xpos[end_effector_index].copy()
# diff = ee1 - ee2



env = wrappers.Kitchen(size=(512, 512),control_version='mocap_ik', task='microwave', simple_randomization=True)
# env._env.render(mode='human')
env.reset()  # [-0.108  0.607  2.6  ]
mocap_index = env.mocap_index
end_effector_index = env.end_effector_index
mocap_pos = env._env.sim.data.site_xpos[mocap_index].copy()
mocap_quat = env._env.sim.data.mocap_quat
while True:

    plt.imshow(env.render())
    plt.show()
    mocap_pos = env._env.sim.data.site_xpos[mocap_index].copy()
    mocap_quat = env._env.sim.data.mocap_quat
    print("MOCAP POS", mocap_pos)
    # print("MOCAP QUAT", mocap_quat)
    # print("MOCAP", env._env.data.mocap_pos)
    print("Endeff pos", env._env.sim.data.site_xpos[end_effector_index].copy())  # CORRECT ONE
    # print("Endeff pos2", env._env.sim.model.site_pos[end_effector_index].copy())
    print("Cabinet pos", env._env.sim.data.site_xpos[env._env.sim.model._site_name2id['cabinet_door']])
    print("Cabinet goal", env._env.sim.data.site_xpos[env._env.sim.model._site_name2id['cabinet_door_goal']])
    env.get_reward()


    var = input('var')
    val = float(input('val'))
    vars = ['x', 'y', 'z']
    # a = float(input('a'))
    # b = float(input('b'))
    # c = float(input('c'))
    # d = float(input('d'))

    var_index = vars.index(var)
    mocap_pos[var_index] = val
    env._env.data.set_mocap_pos('mocap', mocap_pos)
    # print("SET MOCAP", env._env.sim.data.site_xpos[mocap_index].copy())
    # env._env.data.set_mocap_quat('mocap', np.array([a, b, c, d]))

    for _ in range(2000):
        # print("sim step", env._env.sim.data.site_xpos[mocap_index].copy(), env._env.sim.data.site_xpos[end_effector_index].copy())
        try:
            env._env.sim.step()
        except:
            print("????")
#

# env._env.sim.model.body_mass[48:49] = 99
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
