import sqlite3
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy

def getData(data_base, table_name, name_column, cond):
    """
    get data from database
    :param data_base: path of the database
    :param table_name: name of the table
    :param cond: extra condition
    :return:
    """
    metric = ''
    for name in name_column:
        metric += name + ','
    metric=metric[:-1]
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cursor.execute(
        ' SELECT '+ metric +
        ' FROM ' + table_name +
        cond
    )
    data_all = cursor.fetchall()
    if len(data_all) == 0:
        return None
    name_column = [description[0] for description in cursor.description]
    datas = {}
    for id, name in enumerate(name_column):
        datas[name] = []
        if isinstance(data_all[0][id], bytes):
            for data in data_all:
                datas[name].append(struct.unpack('f', data[id])[0])
        else:
            for data in data_all:
                datas[name].append(data[id])

    return datas


def getData_order(data_base, table_name, name_column, cond):
    """
    get data from database
    :param data_base: path of the database
    :param table_name: name of the table
    :param cond: extra condition
    :return:
    """
    metric = ''
    for name in name_column:
        metric += name + ','
    metric=metric[:-1]
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cursor.execute(
        ' SELECT '+ metric + ' FROM ( '+
        ' SELECT * FROM '+ table_name + ' '+
        cond +
        " ORDER BY " + name_column[0] + ')'
        " ORDER BY " + name_column[1]
    )
    data_all = cursor.fetchall()
    if len(data_all) == 0:
        return None
    name_column = [description[0] for description in cursor.description]
    datas = {}
    for id, name in enumerate(name_column):
        datas[name] = []
        if isinstance(data_all[0][id], bytes):
            for data in data_all:
                datas[name].append(struct.unpack('f', data[id])[0])
        else:
            for data in data_all:
                datas[name].append(data[id])

    return datas

def get_range_exploration(database, table, parameter_name):
    data = getData(database, table, parameter_name, cond='')
    range_parameters = {}
    for name in parameter_name:
        range_parameters[name] = np.unique(data[name])
    return range_parameters



def plot_metric_3d(fig, ax, database, table, name_metric, sweep_params, cond='',
                   steps=16, imshow_range=None, zlog=False):
    data = getData(database, table, [name_metric] + sweep_params, cond)

    # New array containing only those rows where the sweep_params change.
    x_arr = data[sweep_params[0]]
    x_arr_uniq = np.unique(x_arr)
    x_index_label = [x_arr_uniq[0],  x_arr_uniq[x_arr_uniq.shape[0]-1]]
    y_arr = data[sweep_params[1]]
    y_arr_uniq = np.unique(y_arr)
    y_index_label = [y_arr_uniq[0], y_arr_uniq[y_arr_uniq.shape[0]-1]]
    if zlog:
        z_arr = np.log10(data[sweep_params[2]])
    else:
        z_arr = data[sweep_params[2]]
    z_arr_uniq = np.unique(z_arr)
    z_index_label = [z_arr_uniq[0], z_arr_uniq[z_arr_uniq.shape[0]-1]]
    c_arr = np.array(data[name_metric])
    if len(np.where(c_arr == None)[0]) != 0:
        c_arr[np.where(c_arr == None)] = np.NAN
        c_arr = np.array(c_arr, dtype=float)

    # Make a nice title
    title = "%s"%(name_metric)

    if type(imshow_range) is type(None):
        imshow_range = (None, None)
    # plot the image and manage the axis
    img = ax.scatter(x_arr, y_arr, z_arr, c=c_arr, cmap=plt.plasma(), vmin=imshow_range[0], vmax=imshow_range[1])
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    ax.set_zticks(ticks=[])
    # ax.set_xticks(ticks=x_index_label)
    # ax.set_xticklabels(labels=np.around(x_index_label, decimals=2))
    # ax.set_yticks(ticks=y_index_label)
    # ax.set_yticklabels(labels=np.around(y_index_label, decimals=2))
    # ax.set_zticks(ticks=z_index_label)
    # ax.set_zticklabels(labels=np.around(z_index_label, decimals=2))
    # ax.set_xlabel(sweep_params[0], labelpad=8)
    # ax.set_ylabel(sweep_params[1], labelpad=8)
    # ax.set_zlabel(sweep_params[2], labelpad=8)
    ax.set_title(title)


    plt.tight_layout()
    fig.colorbar(img, pad=0.1)

def plot_metric_2d(fig, ax, database, table, name_metric, sweep_params, cond='',
                   steps=16, imshow_range=None):
    data = getData_order(database, table, sweep_params + [name_metric], cond)

    # New array containing only those rows where the sweep_params change.
    x_arr = np.unique(data[sweep_params[0]])
    x_index_label = [0, int(x_arr.shape[0]/2), x_arr.shape[0]-1]
    y_arr = np.unique(data[sweep_params[1]])
    y_index_label = [0, int(y_arr.shape[0]/2), y_arr.shape[0]-1]
    c_arr = np.array(data[name_metric]).reshape((y_arr.shape[0], x_arr.shape[0]))
    assert np.unique(np.array(data[sweep_params[1]]).reshape((y_arr.shape[0], x_arr.shape[0])), axis=1).shape[1] == 1
    assert np.unique(np.array(data[sweep_params[0]]).reshape((y_arr.shape[0], x_arr.shape[0])), axis=0).shape[0] == 1
    if len(np.where(c_arr == None)[0]) != 0:
        c_arr[np.where(c_arr == None)] = np.NAN
        c_arr = np.array(c_arr, dtype=float)

    # Make a nice title
    title = "%s"%(name_metric)

    if type(imshow_range) is type(None):
        imshow_range = (None, None)
    # plot the image and manage the axis
    cmap = copy.copy(mpl.cm.get_cmap("plasma"))
    cmap.set_bad(color='white')
    img = ax.imshow(c_arr, cmap=cmap, vmin=imshow_range[0], vmax=imshow_range[1],origin='lower')
    ax.set_xticks(ticks=x_index_label)
    ax.set_xticklabels(labels=np.around(x_arr[x_index_label], decimals=2))
    ax.set_yticks(ticks=y_index_label)
    ax.set_yticklabels(labels=np.around(y_arr[y_index_label], decimals=2))
    ax.set_xlabel(sweep_params[0], labelpad=8)
    ax.set_ylabel(sweep_params[1], labelpad=8)
    ax.set_title(title)
    fig.colorbar(img, pad=0.1)


def plot_exploration(database, table, parameter_display, parameter_sequence,
                     param_exploration_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in'],
                     name_measures_display=['mean_FR_e'],
                     list_choice=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     ):
    range_exploration = get_range_exploration(database, table, param_exploration_name)
    index_sequence_param = np.where([i == parameter_sequence for i in param_exploration_name])[0][0]
    figures = []
    for index_sequence in range(len(range_exploration[parameter_sequence])):
        title = param_exploration_name[index_sequence_param]+':'+str(range_exploration[parameter_sequence][index_sequence])
        list_choice[index_sequence_param] = index_sequence
        param_choice = {}
        for name, index_value in zip(param_exploration_name, list_choice):
            param_choice[name] = range_exploration[name][index_value]
        cond = " WHERE "
        for name in param_choice.keys():
            if not (name in parameter_display):
                cond += name + " = " + str(param_choice[name]) + " AND "
        cond = cond[:-4]
        nb_graph = np.ceil(np.sqrt(len(name_measures_display)))
        fig = plt.figure(figsize=(20, 20))
        for index, name_measure in enumerate(name_measures_display):
            print(name_measure)
            if len(parameter_display) == 2:
                ax = fig.add_subplot(nb_graph, nb_graph, index + 1)
                plot_metric_2d(fig, ax, database, table, name_measure, parameter_display, cond=cond)
            elif len(parameter_display) ==3:
                ax = fig.add_subplot(nb_graph, nb_graph, index + 1, projection='3d')
                plot_metric_3d(fig, ax, database, table, name_measure, parameter_display, cond=cond, zlog=(parameter_display[2]=='wNoise'))
        if len(parameter_display) == 2:
            plt.suptitle(title+' exploration %s %s'%(parameter_display[0], parameter_display[1]))
            plt.subplots_adjust(top=0.945, bottom=0.022, left=0.00, right=0.979, hspace=0.369)
        elif len(parameter_display) == 3:
            plt.suptitle(title +' exploration %s %s %s'%(parameter_display[0], parameter_display[1], parameter_display[2]))
            plt.subplots_adjust(top=0.91, bottom=0.0, left=0.00, right=0.90, hspace=0.3)
        figures.append(fig)
    return figures
