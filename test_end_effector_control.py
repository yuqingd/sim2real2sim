import matplotlib
# matplotlib.use('Qt5Agg')

import numpy as np
import wrappers
import moviepy.editor as mpy
from matplotlib import pyplot as plt

env = wrappers.Kitchen(size=(512, 512), step_repeat=50, control_version='mocap_ik')
env.reset()
mocap_index = env.mocap_index
end_effector_index = env.end_effector_index
mocap_pos = env._env.sim.data.site_xpos[mocap_index].copy()
mocap_quat = env._env.sim.data.mocap_quat
print("MOCAP POS", mocap_pos)
print("Endeff pos", env._env.sim.data.site_xpos[end_effector_index].copy())

# while True:
#     plt.imshow(env.render())
#     plt.show()
#     mocap_pos = env._env.sim.data.site_xpos[mocap_index].copy()
#     mocap_quat = env._env.sim.data.mocap_quat
#     print("MOCAP POS", mocap_pos)
#     print("MOCAP QUAT", mocap_quat)
#     print("MOCAP", env._env.data.mocap_pos)
#     print("Endeff pos", env._env.sim.data.site_xpos[end_effector_index].copy())
#     var = input('var')
#     val = float(input('val'))
#     vars = ['x', 'y', 'z']
#     a = float(input('a'))
#     b = float(input('b'))
#     c = float(input('c'))
#     d = float(input('d'))
#
#     var_index = vars.index(var)
#     mocap_pos[var_index] = val
#     env._env.data.set_mocap_pos('mocap', mocap_pos)
#     print("SET MOCAP", env._env.sim.data.site_xpos[mocap_index].copy())
#     env._env.data.set_mocap_quat('mocap', np.array([a, b, c, d]))
#
#     for _ in range(2000):
#         # print("sim step", env._env.sim.data.site_xpos[mocap_index].copy(), env._env.sim.data.site_xpos[end_effector_index].copy())
#         try:
#             env._env.sim.step()
#         except:
#             print("????")




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
    # a[0] = -1
    # a[i] = 1  # Specify a change along one axis. We could also comment this out to check that with no change the arm stays still.
    for k in range(35):
        o, _, _, _ = env.step(a)
        frames.append(env.render(size=(img_size,img_size)).copy())
        positions.append(env._env.sim.data.site_xpos[end_effector_index].copy())
    fps = 10
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
