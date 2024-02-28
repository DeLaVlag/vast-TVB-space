import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from copy import copy

from helper import getData_order, get_range_exploration

database = "../data/mGPU_TVB.db"

table = 'exploration'
labelticks_size = 10
labelticks_legend= 8
ticks_size = 10
linewidth = 1.0
nb_GPU = 11
nb_best = 1

sweep_params = ['g', 'be', 'ex_ex', 'in_ex']
measures = {'mean_FR_e':{'max':5, 'min':0.0, 'unit':'excitatory\nmean\nfiring rate\nin Hz'},
            'std_FR_e':{'max':2.5, 'min':0.5, 'unit':'excitatory\nvariance\nfiring rate\nin Hz'},
            'fmax_prom_e':{'max':20, 'min':0.0, 'unit':'excitatory\ndominant\nfrequency\nin Hz'},
            'corr_FC_SC_e':{'max':0.5, 'min':0.0, 'unit':'excitatory\ncorrelation \n FS SC'},
            }

param_exploration_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']
range_exploration = get_range_exploration(database, table, param_exploration_name)
print(get_range_exploration(database, table, param_exploration_name))
list_choice = [0,0,3,1,1,1,0,0,0,0]
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
z_arr = np.concatenate((np.unique(data[sweep_params[2]]), np.unique(data[sweep_params[3]])))
z_index_label = z_arr

data_measure = []
for name in measures.keys() :
    c_arr = np.array(data[name]).reshape((y_arr.shape[0], x_arr.shape[0], z_arr.shape[0]))
    assert np.unique(np.array(data[sweep_params[0]]).reshape((y_arr.shape[0], x_arr.shape[0], z_arr.shape[0]))[:,:,0], axis=0).shape[0] == 1
    assert np.unique(np.array(data[sweep_params[1]]).reshape((y_arr.shape[0], x_arr.shape[0], z_arr.shape[0]))[:,0,:], axis=1).shape[1] == 1
    assert np.unique(np.array(data[sweep_params[2]]).reshape((y_arr.shape[0], x_arr.shape[0], z_arr.shape[0]))[0,:,:], axis=0).shape[0] == 1
    if len(np.where(c_arr == None)[0]) != 0:
        c_arr[np.where(c_arr == None)] = np.NAN
        c_arr = np.array(c_arr, dtype=float)
    data_measure.append(c_arr)

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

fig, axs = plt.subplots(4,4,figsize=(6.8, 6.8))
for j in range(4):
    for index, name in enumerate(measures.keys()):
        # plot the image and manage the axis
        cmap = copy(mpl.cm.get_cmap("plasma"))
        cmap.set_bad(color='white')
        ax = axs[index, j]
        img = ax.imshow(data_measure[index][:,:,j], cmap=cmap, vmin=measures[name]['min'], vmax=measures[name]['max'], origin='lower')
        if j == 3:
            # aspect = 10
            # pad_fraction = 1.0
            # divider = make_axes_locatable(ax)
            # width = axes_size.AxesY(ax, aspect=1./aspect)
            # pad = axes_size.Fraction(pad_fraction, width)
            # cax = divider.append_axes("right", size=width, pad=pad)
            # ax2 = ax.twinx()
            # ax2.get_yaxis().set_visible(False)
            # ax2.get_xaxis().set_visible(False)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            if index == 0:
                cax = plt.axes([0.84, 0.745, 0.015, 0.18])
            elif index == 1:
                cax = plt.axes([0.84, 0.52, 0.015, 0.18])
            elif index == 2:
                cax = plt.axes([0.84, 0.29, 0.015, 0.18])
            elif index == 3:
                cax = plt.axes([0.84, 0.065, 0.015, 0.18])
            cl = fig.colorbar(img, cax=cax, orientation='vertical')
            cl.set_ticks(ticks=[measures[name]['min'],
                                measures[name]['min'] + (measures[name]['max'] - measures[name]['min']) / 2,
                                measures[name]['max']])
            cl.ax.set_ylabel(measures[name]['unit'])
            cl.ax.tick_params(axis='both', labelsize=ticks_size)
        if j == 0:
            ax.set_ylabel(sweep_params[1], {"fontsize": labelticks_size}, labelpad=2)
            ax.set_yticks(ticks=y_index_label)
            ax.set_yticklabels(labels=np.around(y_arr[y_index_label], decimals=2))
        else:
            ax.get_yaxis().set_visible(False)
        if index == 3:
            ax.set_xticks(ticks=x_index_label)
            ax.set_xticklabels(labels=np.around(x_arr[x_index_label], decimals=2))
            ax.set_xlabel(sweep_params[0], {"fontsize": labelticks_size}, labelpad=2)
        else:
            ax.get_xaxis().set_visible(False)
        ax.tick_params(axis='both', labelsize=ticks_size)
        if index ==0:
            ax.set_title(r'$C_F_e,e:$'+str(np.around(z_arr[j//2], decimals=5))+"\n"
                         +r'$C_F_i,e:$'+str(np.around(z_arr[j%2], decimals=5)))
plt.subplots_adjust(left=0.105, bottom=0.04, top=0.95, right=0.82, wspace=0.055, hspace=0.0)
# plt.show()
plt.savefig("ex.pdf", dpi=300)


