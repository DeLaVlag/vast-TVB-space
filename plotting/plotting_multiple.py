'''
Script for plotting the multiple graphs (figure 2) for the "Vast TVB parameter space exploration: A Modular Framework
for Accelerating the Multi-Scale Simulation of Human Brain Dynamics" manuscript
'''


import matplotlib.pyplot as plt
import numpy as np
import itertools
from IPython.display import clear_output

import pickle

tavg_data = np.load("./data/b10_FC-4_0-120_g3-8.npy", allow_pickle=True)


#mooi
print(tavg_data.shape)

FR_exc = []
FR_inh = []
Ad_exc = []

n_params = 36
labelticks_size = 12

fig, axes = plt.subplots(6, 6, figsize=(16, 8))

# time_s from ms to sec
time_s = range(2000, len(tavg_data[2000:, 0, 0] * 1e-3)+2000)

#noiseloop
# for h in range(8):
for i in range(0, 6):

    plt.rcParams.update({'font.size': 14})

    slh = [0.3, 0.8, "{:.0f}".format(0), "{:.0f}".format(200), -63.0, -63.0, -65.0, -65.0, 40.0, 40.0]
    s0 = np.linspace(slh[0], slh[1], 6)
    s1 = np.linspace(int(slh[2]), int(slh[3]), 6)

    params = itertools.product(s0, s1)
    params = np.array([vals for vals in params], np.float32)


    for j in range(0, 6):
        '''fill variables original'''
        FR_exc = tavg_data[2000:, 0, :, i*6+j] * 1e3  # from KHz to Hz; Excitatory firing rate
        FR_inh = tavg_data[2000:, 1, :, i*6+j] * 1e3  # from KHz to Hz; Inhibitory firing rate

        '''plot traces'''
        Le = axes[i, j].plot(time_s, FR_inh, color='SteelBlue', alpha=.1)  # [times, regions]
        Li = axes[i, j].plot(time_s, FR_exc, color='darkred')  # [times, regions]

        axes[i, j].tick_params(axis='x', labelsize=labelticks_size)
        axes[i, j].tick_params(axis='y', labelsize=labelticks_size)

        axes[5, j].set_xlabel('Time (ms)', {"fontsize": labelticks_size})

        axes[i, 0].set_ylabel('F. rate (Hz)', {"fontsize": labelticks_size})

        axes[i, j].set_title('[g, b_e]:', loc='left', fontsize=labelticks_size)
        axes[i, j].set_title(np.array2string(params[i*6+j], separator=', '), fontsize=labelticks_size)

plt.tight_layout()
plt.show()

