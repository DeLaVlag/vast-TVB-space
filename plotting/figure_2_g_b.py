import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy

from helper import getData_order, get_range_exploration

database = "../data/mGPU_TVB.db"

table = 'exploration'
labelticks_size = 12
labelticks_legend= 8
ticks_size = 12
linewidth = 1.0
nb_GPU = 11
nb_best = 1

sweep_params = ['g', 'be']
measures = {'mean_FR_e':{'max':5, 'min':0.0, 'unit':'excitatory mean\nfiring rate in Hz'},
            'mean_FR_i':{'max':20, 'min':0.0, 'unit':'inhibitory mean\nfiring rate in Hz'},
            'std_FR_e':{'max':2, 'min':0.0, 'unit':'excitatory variance\nfiring rate in Hz'},
            'std_FR_i':{'max':5, 'min':2.0, 'unit':'inhibitory variance\nfiring rate in Hz'},}

param_exploration_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']
range_exploration = get_range_exploration(database, table, param_exploration_name)
print(get_range_exploration(database, table, param_exploration_name))
list_choice = [0,0,3,1,1,0,0,0,0]
param_choice = {}
for name, index_value in zip(param_exploration_name, list_choice):
    param_choice[name] = range_exploration[name][index_value]
cond = " WHERE "
for name in param_choice.keys():
    if not (name in sweep_params):
        cond += name + " = " + str(param_choice[name]) + " AND "
cond = cond[:-4]
print(cond)
data = getData_order(database, table, sweep_params + list(measures.keys()), cond)
# New array containing only those rows where the sweep_params change.
x_arr = np.unique(data[sweep_params[0]])
x_index_label = [0, int(x_arr.shape[0] / 2), x_arr.shape[0] - 1]
y_arr = np.unique(data[sweep_params[1]])
y_index_label = [0, int(y_arr.shape[0] / 2), y_arr.shape[0] - 1]
data_measure = []
for name in measures.keys() :
    c_arr = np.array(data[name]).reshape((y_arr.shape[0], x_arr.shape[0]))
    assert np.unique(np.array(data[sweep_params[1]]).reshape((y_arr.shape[0], x_arr.shape[0])), axis=1).shape[1] == 1
    assert np.unique(np.array(data[sweep_params[0]]).reshape((y_arr.shape[0], x_arr.shape[0])), axis=0).shape[0] == 1
    if len(np.where(c_arr == None)[0]) != 0:
        c_arr[np.where(c_arr == None)] = np.NAN
        c_arr = np.array(c_arr, dtype=float)
    data_measure.append(c_arr)

fig, axs = plt.subplots(2,2,figsize=(9, 3.6))

for index, name in enumerate(measures.keys()):
    # plot the image and manage the axis
    cmap = copy(mpl.cm.get_cmap("plasma"))
    cmap.set_bad(color='white')
    ax = axs[int(index/2), index%2]
    img = ax.imshow(data_measure[index], cmap=cmap, vmin=measures[name]['min'], vmax=measures[name]['max'], origin='lower')
    cl = fig.colorbar(img, ax=ax)
    cl.set_ticks(ticks=[measures[name]['min'], measures[name]['min']+(measures[name]['max']-measures[name]['min'])/2, measures[name]['max']])
    cl.ax.set_ylabel(measures[name]['unit'], {"fontsize": labelticks_size})
    cl.ax.tick_params(axis='both', labelsize=ticks_size)
    ax.set_xticks(ticks=x_index_label)
    ax.set_xticklabels(labels=np.around(x_arr[x_index_label], decimals=2))
    ax.set_yticks(ticks=y_index_label)
    ax.set_yticklabels(labels=np.around(y_arr[y_index_label], decimals=2))
    if index %2 ==0:
        ax.set_ylabel(sweep_params[1], {"fontsize": labelticks_size}, labelpad=2)
    if index // 2 :
        ax.set_xlabel(sweep_params[0], {"fontsize": labelticks_size}, labelpad=2)
    ax.tick_params(axis='both', labelsize=ticks_size)
plt.subplots_adjust(left=0.0, bottom=0.125, top=0.98, right=0.95)
plt.savefig("g_b_e.pdf", dpi=300)


