import matplotlib
# matplotlib.use('Qt5Agg')

import numpy as np
import wrappers
import moviepy.editor as mpy
from matplotlib import pyplot as plt

env = wrappers.Kitchen(size=(256, 256), step_repeat=200, control_version='mocap_ik')

# ====================== check whether end_effector control works ================

# Axis aligned
img_size = 512
end_effector_index = env.end_effector_index
for i in range(3):
    print("TEST", i)
    o = env.reset()
    offset = np.zeros((3,))
    offset[i] = 1  # Will be scaled down by the step size
    frames = [env.render(size=(img_size,img_size))]
    positions = [env._env.sim.data.site_xpos[end_effector_index].copy()]
    a = np.zeros((3,))
    a[i] = 1  # Specify a change along one axis. We could also comment this out to check that with no change the arm stays still.
    for k in range(100):
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
