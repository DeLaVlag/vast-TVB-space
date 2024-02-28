import numpy as np
import matplotlib.pyplot as plt
from tvb.datatypes.connectivity import Connectivity

# Generate random colormap
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True, seed=0):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap, CSS4_COLORS, rgb_to_hsv, to_rgb
    np.random.seed(seed)
    # Sort colors by hue, saturation, value and name.
    names = np.array(sorted(CSS4_COLORS, key=lambda c: tuple(to_rgb(c))))
    colors = []
    while len(colors) < nlabels:
        name = names[np.random.randint(0, len(CSS4_COLORS))]
        color = CSS4_COLORS[name]
        if rgb_to_hsv(to_rgb(CSS4_COLORS[name]))[2] < 0.97 and rgb_to_hsv(to_rgb(CSS4_COLORS[name]))[2] != 0.0\
                and color not in colors:
            colors.append(color)
    random_colormap = LinearSegmentedColormap.from_list('new_map', colors, N=nlabels)

    return random_colormap


path = "../data/tavg_b10.npy"
con = Connectivity().from_file("../input/connectivity_68.zip")

labelticks_size = 10
labelticks_legend = 8
ticks_size = 10
linewidth = .9

nb_GPU = 10
nb_best = 13 #1,3,13

data_all = np.load(path, allow_pickle=True)[nb_GPU]
data = data_all[nb_best]
parameters = {}
for i in ['time', 'g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']:
    parameters[i] = data[i]
print(parameters)

times_serie = data['TS']
print(np.mean(times_serie[:, 0, :]), np.max(times_serie[:, 0, :] * 1e3))

new_cmap = rand_cmap(68, type='bright', first_color_black=True, last_color_black=False, verbose=True)
colors = [new_cmap(i) for i in np.linspace(0, 1, 68)]

plt.figure(figsize=(6.8, 6.8))
for index, ts in enumerate(times_serie[:, 0, :].T * 1e3):
    plt.plot(ts, linewidth=linewidth, label=con.region_labels[index], c=colors[index])
plt.legend(bbox_to_anchor=(1.01, 0., 0.0, 1.0), fontsize=labelticks_legend, ncol=2,
           borderpad=0.2, labelspacing=0.7, handletextpad=0.2, borderaxespad=0.2, columnspacing=0.5)
plt.xlabel('time in ms', {"fontsize": labelticks_size})
plt.xlim(xmax=4000.0, xmin=-0.1)
plt.ylabel('frequency of excitatory population in Hz', {"fontsize": labelticks_size})
plt.ylim(ymax=200.0, ymin=-0.1)
plt.tick_params(axis='both', labelsize=ticks_size)
plt.subplots_adjust(left=0.09, right=0.47, top=0.995, bottom=0.065)
# plt.show()
plt.savefig("timeserie_epileptisie.pdf", dpi=300)

