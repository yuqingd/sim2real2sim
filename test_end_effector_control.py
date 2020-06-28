import matplotlib
# matplotlib.use('Qt5Agg')

import numpy as np
import wrappers
import moviepy.editor as mpy
from matplotlib import pyplot as plt

# env = wrappers.MetaWorld("reach", size=(512, 512))
# env = wrappers.Kitchen(control_version='metaworld_ik', step_size=0.1)
# env.reset()
# while True:
#     low = env.action_space.low
#     high = env.action_space.high
#     # a = env.action_space.sample(low)
#     a = np.zeros((3,))
#     a[2] = -1
#     env.step(a)
#     o = env.render(mode='rgb_array')
#     # plt.imshow(o)
#     plt.imshow(env.render(size=(512, 512)))
#     plt.show()



# env = wrappers.Kitchen(control_version='metaworld_ik')
# env.reset()
# x = env.action_space
# while True:
#     img = env._env.sim.render(width=512, height=512, camera_name="top_camera")
#     print()
#     plt.imshow(img)
#     plt.show()
#     a = env.action_space.sample()
#     env.step(a)


# while True:
#     distance = float(input('distance'))
#     azimuth = float(input('azimuth'))
#     elevation = float(input('elevation'))
#
#     env = wrappers.Kitchen(control_version='metaworld_ik', distance=distance, azimuth=azimuth, elevation=elevation)
#     env.reset()
#     o = env.render(size=(512, 512))
#     plt.imshow(o)
#     plt.show()
#     print("??")



env = wrappers.Kitchen(size=(256, 256), step_repeat=100, control_version='metaworld_ik')
# from dm_control.utils.inverse_kinematics import qpos_from_site_pose
# end_effector_index = env.end_effector_index
# mocap_index = env.mocap_index
# orig_end_effector = env._env.sim.data.site_xpos[end_effector_index].copy()
# orig_mocap = env._env.sim.data.site_xpos[mocap_index].copy()
# orig_mocap2 = env._env.data.mocap_pos.copy()
# a = np.zeros((3,))
# a[0] = -1
# env.step(a)
# final_end_effector = env._env.sim.data.site_xpos[end_effector_index].copy()
# final_mocap = env._env.sim.data.site_xpos[mocap_index].copy()
# final_mocap2 = env._env.data.mocap_pos.copy()

# ====================== check whether end_effector control works ================

# k = env._env.sim.named
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
    # a[i-1] = 20
    a[i] = -1  # Specify a change along one axis. We could also comment this out to check that with no change the arm stays still.
    #xyz_diff = a * env.step_size
    #a = env._env.sim.data.site_xpos[env.end_effector_index] + xyz_diff
    for k in range(1000):
        o, _, _, _ = env.step(a)
        # env._env.step(np.zeros((13,)))
        # plt.imshow(env._env.render(mode='rgb_array')):q:qq
        # plt.show()
        if k % 10 == 0:
            frames.append(env.render(size=(img_size,img_size)).copy())
            positions.append(env._env.sim.data.site_xpos[end_effector_index].copy())
    fps = 20
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_gif('test_end_effector_axis' + str(i) + '.gif', fps=fps)
    # We'd expect to see a graph where all positions stay the same except the one along which we're moving.
    for j in range(3):
        data = [p[j] for p in positions]
        plt.plot(range(len(data)), data)
    plt.legend(['x', 'y', 'z'])
    plt.savefig('plot_end_effector_axis' + str(i) + '.png')
    plt.show()
    plt.close()

assert False
# ==============================================================================================


import wrappers
import numpy as np
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
env = wrappers.KitchenTaskRelaxV1()
obs_prev = env.reset()
o_prev = env.render('rgb_array')

from matplotlib import pyplot as plt


orig_qpos = env.sim.data.qpos

# Position it's trying to reach.  This is basically the position it's already at, so we'd expect to see very low movements.
pos = np.array([ 0.369,    -0.57,      2.02])
physics = env.sim
end_effector = 'end_effector'

# copy in joint limits so we can check whether they're out of bounds
qpos_low = np.array([-6.28318530718, -2.05949, -6.28318530718, -0.191986, -6.28318530718, -1.69297, -6.28318530718, 0, 0, 0, 0, 0, 0])
qpos_high = np.array([6.28318530718, 2.0944, 6.28318530718, 3.92699, 6.28318530718, 3.14159, 6.28318530718, .85, .85, .85, .85, .85, .85])

# Move to the right location
while True:

    # find a reasonable set of constraints
    while True:
        # use IK to find desired qpos
        # TODO: is the right way to be using qpos_from_site_pos
        qvel = env.sim.data.qpos
        qvel_max = np.max(np.abs(qvel))
        qvel0 = qvel[0]
        ikresult = qpos_from_site_pose(physics, end_effector, target_pos=pos)
        qpos = ikresult.qpos
        success = ikresult.success
        max_v = np.max(np.abs(qpos))

        # Check whether the proposed positions are in range
        high_pass = (qpos[:13] <= qpos_high).all()
        low_pass = (qpos[:13] >= qpos_low).all()
        diff0 = qpos - orig_qpos
        diff_const = diff0[13:]
        biggest = np.max(np.abs(diff_const))

        # clip the positions to the valid range before updating # TODO: is this valid to expect that we'd still see the position get closer, not farther away?
        update = np.clip(qpos[:13], qpos_low, qpos_high)
        diff = qpos[:13] - update

        # If we succeed, actually update the sim to go toward this position
        # TODO: are we doing this right?
        if success:
            env.data.ctrl[:] = update
            env.sim.forward()
            env.sim.step()
            break

        diff_again = env.sim.data.qpos - orig_qpos

    # if we get a valid set of joints within the correct joint limits, then (in theory) if we ever get here, we should
    # be able to look at the current qpos and check that it matches the orig_qpos
    if success and high_pass and low_pass:
        break

qpos_final = env.sim.data.qpos
diff = qpos_final - orig_qpos






