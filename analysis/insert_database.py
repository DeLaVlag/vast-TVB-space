import sqlite3
import numpy as np
import sys


def type_database(variable):
    if hasattr(variable, 'dtype'):
        if np.issubdtype(variable, int):
            return 'INTEGER'
        elif np.issubdtype(variable, float):
            return 'REAL'
        else:
            sys.stderr.write('ERROR bad type of save variable\n')
            exit(1)
    else:
        if isinstance(variable, int):
            return 'INTEGER'
        elif isinstance(variable, float):
            return 'REAL'
        elif isinstance(variable, str):
            return 'TEXT'
        else:
            sys.stderr.write('ERROR bad type of save variable\n')
            exit(1)

def init_database(data_base, table_name):
    """
    Initialise the connection to the database et create the table
    :param data_base: file where is the database
    :param table_name: the name of the table
    :return: the connexion to the database
    """
    key_variable = 'g,be,wNoise,speed,tau_w_e,a_e,ex_ex,ex_in,in_ex,in_in'
    measures_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in', 'mean_FR_e_0', 'mean_FR_e_1', 'mean_FR_e_2', 'mean_FR_e_3', 'mean_FR_e_4', 'mean_FR_e_5', 'mean_FR_e_6', 'mean_FR_e_7', 'mean_FR_e_8', 'mean_FR_e_9', 'mean_FR_e_10', 'mean_FR_e_11', 'mean_FR_e_12', 'mean_FR_e_13', 'mean_FR_e_14', 'mean_FR_e_15', 'mean_FR_e_16', 'mean_FR_e_17', 'mean_FR_e_18', 'mean_FR_e_19', 'mean_FR_e_20', 'mean_FR_e_21', 'mean_FR_e_22', 'mean_FR_e_23', 'mean_FR_e_24', 'mean_FR_e_25', 'mean_FR_e_26', 'mean_FR_e_27', 'mean_FR_e_28', 'mean_FR_e_29', 'mean_FR_e_30', 'mean_FR_e_31', 'mean_FR_e_32', 'mean_FR_e_33', 'mean_FR_e_34', 'mean_FR_e_35', 'mean_FR_e_36', 'mean_FR_e_37', 'mean_FR_e_38', 'mean_FR_e_39', 'mean_FR_e_40', 'mean_FR_e_41', 'mean_FR_e_42', 'mean_FR_e_43', 'mean_FR_e_44', 'mean_FR_e_45', 'mean_FR_e_46', 'mean_FR_e_47', 'mean_FR_e_48', 'mean_FR_e_49', 'mean_FR_e_50', 'mean_FR_e_51', 'mean_FR_e_52', 'mean_FR_e_53', 'mean_FR_e_54', 'mean_FR_e_55', 'mean_FR_e_56', 'mean_FR_e_57', 'mean_FR_e_58', 'mean_FR_e_59', 'mean_FR_e_60', 'mean_FR_e_61', 'mean_FR_e_62', 'mean_FR_e_63', 'mean_FR_e_64', 'mean_FR_e_65', 'mean_FR_e_66', 'mean_FR_e_67', 'mean_FR_e', 'std_FR_e', 'coeff_var_e', 'max_FR_e_0', 'max_FR_e_1', 'max_FR_e_2', 'max_FR_e_3', 'max_FR_e_4', 'max_FR_e_5', 'max_FR_e_6', 'max_FR_e_7', 'max_FR_e_8', 'max_FR_e_9', 'max_FR_e_10', 'max_FR_e_11', 'max_FR_e_12', 'max_FR_e_13', 'max_FR_e_14', 'max_FR_e_15', 'max_FR_e_16', 'max_FR_e_17', 'max_FR_e_18', 'max_FR_e_19', 'max_FR_e_20', 'max_FR_e_21', 'max_FR_e_22', 'max_FR_e_23', 'max_FR_e_24', 'max_FR_e_25', 'max_FR_e_26', 'max_FR_e_27', 'max_FR_e_28', 'max_FR_e_29', 'max_FR_e_30', 'max_FR_e_31', 'max_FR_e_32', 'max_FR_e_33', 'max_FR_e_34', 'max_FR_e_35', 'max_FR_e_36', 'max_FR_e_37', 'max_FR_e_38', 'max_FR_e_39', 'max_FR_e_40', 'max_FR_e_41', 'max_FR_e_42', 'max_FR_e_43', 'max_FR_e_44', 'max_FR_e_45', 'max_FR_e_46', 'max_FR_e_47', 'max_FR_e_48', 'max_FR_e_49', 'max_FR_e_50', 'max_FR_e_51', 'max_FR_e_52', 'max_FR_e_53', 'max_FR_e_54', 'max_FR_e_55', 'max_FR_e_56', 'max_FR_e_57', 'max_FR_e_58', 'max_FR_e_59', 'max_FR_e_60', 'max_FR_e_61', 'max_FR_e_62', 'max_FR_e_63', 'max_FR_e_64', 'max_FR_e_65', 'max_FR_e_66', 'max_FR_e_67', 'max_FR_e', 'min_FR_e_0', 'min_FR_e_1', 'min_FR_e_2', 'min_FR_e_3', 'min_FR_e_4', 'min_FR_e_5', 'min_FR_e_6', 'min_FR_e_7', 'min_FR_e_8', 'min_FR_e_9', 'min_FR_e_10', 'min_FR_e_11', 'min_FR_e_12', 'min_FR_e_13', 'min_FR_e_14', 'min_FR_e_15', 'min_FR_e_16', 'min_FR_e_17', 'min_FR_e_18', 'min_FR_e_19', 'min_FR_e_20', 'min_FR_e_21', 'min_FR_e_22', 'min_FR_e_23', 'min_FR_e_24', 'min_FR_e_25', 'min_FR_e_26', 'min_FR_e_27', 'min_FR_e_28', 'min_FR_e_29', 'min_FR_e_30', 'min_FR_e_31', 'min_FR_e_32', 'min_FR_e_33', 'min_FR_e_34', 'min_FR_e_35', 'min_FR_e_36', 'min_FR_e_37', 'min_FR_e_38', 'min_FR_e_39', 'min_FR_e_40', 'min_FR_e_41', 'min_FR_e_42', 'min_FR_e_43', 'min_FR_e_44', 'min_FR_e_45', 'min_FR_e_46', 'min_FR_e_47', 'min_FR_e_48', 'min_FR_e_49', 'min_FR_e_50', 'min_FR_e_51', 'min_FR_e_52', 'min_FR_e_53', 'min_FR_e_54', 'min_FR_e_55', 'min_FR_e_56', 'min_FR_e_57', 'min_FR_e_58', 'min_FR_e_59', 'min_FR_e_60', 'min_FR_e_61', 'min_FR_e_62', 'min_FR_e_63', 'min_FR_e_64', 'min_FR_e_65', 'min_FR_e_66', 'min_FR_e_67', 'min_FR_e', 'mean_FC_e', 'mean_PLI_e', 'mean_up_e', 'mean_down_e', 'fmax_amp_e', 'pmax_amp_e', 'fmax_prom_e', 'pmax_prom_e', 'freq_dom_e_0', 'freq_dom_e_1', 'freq_dom_e_2', 'freq_dom_e_3', 'freq_dom_e_4', 'freq_dom_e_5', 'freq_dom_e_6', 'freq_dom_e_7', 'freq_dom_e_8', 'freq_dom_e_9', 'freq_dom_e_10', 'freq_dom_e_11', 'freq_dom_e_12', 'freq_dom_e_13', 'freq_dom_e_14', 'freq_dom_e_15', 'freq_dom_e_16', 'freq_dom_e_17', 'freq_dom_e_18', 'freq_dom_e_19', 'freq_dom_e_20', 'freq_dom_e_21', 'freq_dom_e_22', 'freq_dom_e_23', 'freq_dom_e_24', 'freq_dom_e_25', 'freq_dom_e_26', 'freq_dom_e_27', 'freq_dom_e_28', 'freq_dom_e_29', 'freq_dom_e_30', 'freq_dom_e_31', 'freq_dom_e_32', 'freq_dom_e_33', 'freq_dom_e_34', 'freq_dom_e_35', 'freq_dom_e_36', 'freq_dom_e_37', 'freq_dom_e_38', 'freq_dom_e_39', 'freq_dom_e_40', 'freq_dom_e_41', 'freq_dom_e_42', 'freq_dom_e_43', 'freq_dom_e_44', 'freq_dom_e_45', 'freq_dom_e_46', 'freq_dom_e_47', 'freq_dom_e_48', 'freq_dom_e_49', 'freq_dom_e_50', 'freq_dom_e_51', 'freq_dom_e_52', 'freq_dom_e_53', 'freq_dom_e_54', 'freq_dom_e_55', 'freq_dom_e_56', 'freq_dom_e_57', 'freq_dom_e_58', 'freq_dom_e_59', 'freq_dom_e_60', 'freq_dom_e_61', 'freq_dom_e_62', 'freq_dom_e_63', 'freq_dom_e_64', 'freq_dom_e_65', 'freq_dom_e_66', 'freq_dom_e_67', 'slope_PSD_e', 'score_PSD_e', 'delta_rel_p_e', 'theta_rel_p_e', 'alpha_rel_p_e', 'beta_rel_p_e', 'gamma_rel_p_e', 'ratio_frmean_dmn_e', 'ratio_zscore_dmn_e', 'ratio_AI_e', 'varFCD_e', 'corr_FC_SC_e', 'corr_FC_tract_e', 'mean_FR_i_0', 'mean_FR_i_1', 'mean_FR_i_2', 'mean_FR_i_3', 'mean_FR_i_4', 'mean_FR_i_5', 'mean_FR_i_6', 'mean_FR_i_7', 'mean_FR_i_8', 'mean_FR_i_9', 'mean_FR_i_10', 'mean_FR_i_11', 'mean_FR_i_12', 'mean_FR_i_13', 'mean_FR_i_14', 'mean_FR_i_15', 'mean_FR_i_16', 'mean_FR_i_17', 'mean_FR_i_18', 'mean_FR_i_19', 'mean_FR_i_20', 'mean_FR_i_21', 'mean_FR_i_22', 'mean_FR_i_23', 'mean_FR_i_24', 'mean_FR_i_25', 'mean_FR_i_26', 'mean_FR_i_27', 'mean_FR_i_28', 'mean_FR_i_29', 'mean_FR_i_30', 'mean_FR_i_31', 'mean_FR_i_32', 'mean_FR_i_33', 'mean_FR_i_34', 'mean_FR_i_35', 'mean_FR_i_36', 'mean_FR_i_37', 'mean_FR_i_38', 'mean_FR_i_39', 'mean_FR_i_40', 'mean_FR_i_41', 'mean_FR_i_42', 'mean_FR_i_43', 'mean_FR_i_44', 'mean_FR_i_45', 'mean_FR_i_46', 'mean_FR_i_47', 'mean_FR_i_48', 'mean_FR_i_49', 'mean_FR_i_50', 'mean_FR_i_51', 'mean_FR_i_52', 'mean_FR_i_53', 'mean_FR_i_54', 'mean_FR_i_55', 'mean_FR_i_56', 'mean_FR_i_57', 'mean_FR_i_58', 'mean_FR_i_59', 'mean_FR_i_60', 'mean_FR_i_61', 'mean_FR_i_62', 'mean_FR_i_63', 'mean_FR_i_64', 'mean_FR_i_65', 'mean_FR_i_66', 'mean_FR_i_67', 'mean_FR_i', 'std_FR_i', 'coeff_var_i', 'max_FR_i_0', 'max_FR_i_1', 'max_FR_i_2', 'max_FR_i_3', 'max_FR_i_4', 'max_FR_i_5', 'max_FR_i_6', 'max_FR_i_7', 'max_FR_i_8', 'max_FR_i_9', 'max_FR_i_10', 'max_FR_i_11', 'max_FR_i_12', 'max_FR_i_13', 'max_FR_i_14', 'max_FR_i_15', 'max_FR_i_16', 'max_FR_i_17', 'max_FR_i_18', 'max_FR_i_19', 'max_FR_i_20', 'max_FR_i_21', 'max_FR_i_22', 'max_FR_i_23', 'max_FR_i_24', 'max_FR_i_25', 'max_FR_i_26', 'max_FR_i_27', 'max_FR_i_28', 'max_FR_i_29', 'max_FR_i_30', 'max_FR_i_31', 'max_FR_i_32', 'max_FR_i_33', 'max_FR_i_34', 'max_FR_i_35', 'max_FR_i_36', 'max_FR_i_37', 'max_FR_i_38', 'max_FR_i_39', 'max_FR_i_40', 'max_FR_i_41', 'max_FR_i_42', 'max_FR_i_43', 'max_FR_i_44', 'max_FR_i_45', 'max_FR_i_46', 'max_FR_i_47', 'max_FR_i_48', 'max_FR_i_49', 'max_FR_i_50', 'max_FR_i_51', 'max_FR_i_52', 'max_FR_i_53', 'max_FR_i_54', 'max_FR_i_55', 'max_FR_i_56', 'max_FR_i_57', 'max_FR_i_58', 'max_FR_i_59', 'max_FR_i_60', 'max_FR_i_61', 'max_FR_i_62', 'max_FR_i_63', 'max_FR_i_64', 'max_FR_i_65', 'max_FR_i_66', 'max_FR_i_67', 'max_FR_i', 'min_FR_i_0', 'min_FR_i_1', 'min_FR_i_2', 'min_FR_i_3', 'min_FR_i_4', 'min_FR_i_5', 'min_FR_i_6', 'min_FR_i_7', 'min_FR_i_8', 'min_FR_i_9', 'min_FR_i_10', 'min_FR_i_11', 'min_FR_i_12', 'min_FR_i_13', 'min_FR_i_14', 'min_FR_i_15', 'min_FR_i_16', 'min_FR_i_17', 'min_FR_i_18', 'min_FR_i_19', 'min_FR_i_20', 'min_FR_i_21', 'min_FR_i_22', 'min_FR_i_23', 'min_FR_i_24', 'min_FR_i_25', 'min_FR_i_26', 'min_FR_i_27', 'min_FR_i_28', 'min_FR_i_29', 'min_FR_i_30', 'min_FR_i_31', 'min_FR_i_32', 'min_FR_i_33', 'min_FR_i_34', 'min_FR_i_35', 'min_FR_i_36', 'min_FR_i_37', 'min_FR_i_38', 'min_FR_i_39', 'min_FR_i_40', 'min_FR_i_41', 'min_FR_i_42', 'min_FR_i_43', 'min_FR_i_44', 'min_FR_i_45', 'min_FR_i_46', 'min_FR_i_47', 'min_FR_i_48', 'min_FR_i_49', 'min_FR_i_50', 'min_FR_i_51', 'min_FR_i_52', 'min_FR_i_53', 'min_FR_i_54', 'min_FR_i_55', 'min_FR_i_56', 'min_FR_i_57', 'min_FR_i_58', 'min_FR_i_59', 'min_FR_i_60', 'min_FR_i_61', 'min_FR_i_62', 'min_FR_i_63', 'min_FR_i_64', 'min_FR_i_65', 'min_FR_i_66', 'min_FR_i_67', 'min_FR_i', 'mean_FC_i', 'mean_PLI_i', 'mean_up_i', 'mean_down_i', 'fmax_amp_i', 'pmax_amp_i', 'fmax_prom_i', 'pmax_prom_i', 'freq_dom_i_0', 'freq_dom_i_1', 'freq_dom_i_2', 'freq_dom_i_3', 'freq_dom_i_4', 'freq_dom_i_5', 'freq_dom_i_6', 'freq_dom_i_7', 'freq_dom_i_8', 'freq_dom_i_9', 'freq_dom_i_10', 'freq_dom_i_11', 'freq_dom_i_12', 'freq_dom_i_13', 'freq_dom_i_14', 'freq_dom_i_15', 'freq_dom_i_16', 'freq_dom_i_17', 'freq_dom_i_18', 'freq_dom_i_19', 'freq_dom_i_20', 'freq_dom_i_21', 'freq_dom_i_22', 'freq_dom_i_23', 'freq_dom_i_24', 'freq_dom_i_25', 'freq_dom_i_26', 'freq_dom_i_27', 'freq_dom_i_28', 'freq_dom_i_29', 'freq_dom_i_30', 'freq_dom_i_31', 'freq_dom_i_32', 'freq_dom_i_33', 'freq_dom_i_34', 'freq_dom_i_35', 'freq_dom_i_36', 'freq_dom_i_37', 'freq_dom_i_38', 'freq_dom_i_39', 'freq_dom_i_40', 'freq_dom_i_41', 'freq_dom_i_42', 'freq_dom_i_43', 'freq_dom_i_44', 'freq_dom_i_45', 'freq_dom_i_46', 'freq_dom_i_47', 'freq_dom_i_48', 'freq_dom_i_49', 'freq_dom_i_50', 'freq_dom_i_51', 'freq_dom_i_52', 'freq_dom_i_53', 'freq_dom_i_54', 'freq_dom_i_55', 'freq_dom_i_56', 'freq_dom_i_57', 'freq_dom_i_58', 'freq_dom_i_59', 'freq_dom_i_60', 'freq_dom_i_61', 'freq_dom_i_62', 'freq_dom_i_63', 'freq_dom_i_64', 'freq_dom_i_65', 'freq_dom_i_66', 'freq_dom_i_67', 'slope_PSD_i', 'score_PSD_i', 'delta_rel_p_i', 'theta_rel_p_i', 'alpha_rel_p_i', 'beta_rel_p_i', 'gamma_rel_p_i', 'ratio_frmean_dmn_i', 'ratio_zscore_dmn_i', 'ratio_AI_i', 'varFCD_i', 'corr_FC_SC_i', 'corr_FC_tract_i']
    measures = ''
    for name in measures_name:
        measures += name + ' REAL,'

    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=10000)
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS '
                + table_name
                + '(date TIMESTAMP NOT NULL,'
                + measures
                + 'PRIMARY KEY'
                  '('+key_variable+'))'
                )
    cur.close()
    con.close()


def check_already_analyse_database(data_base, table_name, parameter_name, parameters_values):
    """
    Check if the analysis was already perform
    :param data_base: path of the database
    :param table_name: name of the table
    :param result_path: folder to analyse
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cond = ''
    for name, value in zip(parameter_name, parameters_values):
        cond += name + ' = ' + str(value) +' AND '
    cond = cond [:-5]

    cursor.execute("SELECT * FROM "+table_name+" WHERE "+cond)
    check = len(cursor.fetchall()) != 0
    cursor.close()
    con.close()
    return check


def insert_database(data_base, table_name, results):
    """
    Insert some result in the database
    :param data_base: name of database
    :param table_name: the table where insert the value
    :return: nothing
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=1000)
    cur = con.cursor()
    list_data = []
    for result in results:
        list_data.append(tuple(result.values()))
    keys = ','.join(results[0].keys())
    question_marks = ','.join(list('?' * len(results[0].keys())))
    cur.executemany('INSERT INTO ' + table_name + ' (' + keys + ') VALUES (' + question_marks + ')', list_data)
    con.commit()
    cur.close()
    con.close()


if __name__ == '__main__':
    import time
    path_database = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/test_analysis/analysis/test_databse/database.db'
    table = 'exploration_1'
    init_database(path_database, 'exploration_1')

    from tvb.simulator.lab import connectivity
    from analysis import analysis

    path = "/home/kusch/Documents/project/Zerlaut/compare_zerlaut/test_analysis/data/"
    parameter_name = ['g', 'be', 'wNoise', 'T', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']
    # TODO find a way to load them
    parameter_value = np.loadtxt(path+'/../file_gpu/best10_avg_results.txt')
    connectivity = connectivity.Connectivity().from_file(path + "connectivity_zerlaut_68.zip")
    weights_mat = np.array(connectivity.weights)
    tracts_mat = np.array(connectivity.tract_lengths)
    rank = 0
    timeseries = np.load(path + "tavg_data" + str(rank), allow_pickle=True)
    results = []
    for i in range(10):
        tic = time.time()
        result = analysis(parameter_name, parameter_value[i], timeseries[:, :, :, i], weights_mat, tracts_mat)

        # print(result.keys())  # array for table of measure in initialisation
        if not check_already_analyse_database(path_database, table, parameter_name, parameter_value[i]):
            results.append(result)
            print('insert')
        toc = time.time()
        print(tic - toc)
    insert_database(path_database, table, results)


    # from parameter_analyse.zerlaut_oscilation.python_file.analysis.analysis import analysis
    # path_root = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/'
    # database = path_root + "/database_2.db"
    # table_name = "exploration"
    # init_database(database, table_name)
    # for noise in np.arange(1e-9, 1e-8, 1e-9):
    # # for noise in np.arange(1e-8, 1e-7, 1e-8):
    # # for noise in np.arange(0.0, 1e-5, 5e-7):
    #     for frequency in np.concatenate(([1], np.arange(5., 51., 5.))):
    #         path_simulation = path_root + "frequency_" + str(frequency) + "_noise_" + str(noise) + "/"
    #         if not check_already_analyse_database(database, table_name, path_simulation, 'excitatory'):
    #             results = analysis(path_simulation)
    #             insert_database(database, table_name, results)

    