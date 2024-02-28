import sqlite3
import numpy as np
from colorama import Fore, Back, Style
from tvb.datatypes.connectivity import Connectivity

data_base = "../data/mGPU_TVB.db"
table = 'exploration'
con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
connectome = Connectivity().from_file("../input/connectivity_zerlaut_68.zip")
param_exploration_name = ['g', 'be', 'wNoise', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']
param_exploration_name_db = ''
for name in param_exploration_name:
    param_exploration_name_db += name + ','
param_exploration_name_db = param_exploration_name_db[:-1]

# check which cases if max is higher than  100Hz
print(Fore.RED + "Analyse speed = 1ms" + Fore.RESET)
column = ''
for i in range(68):
    column += 'max_FR_e_' + str(i) + ','
column = column[:-1]

cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE max_FR_e>100.0 AND speed == 1.0'
)
total_number = cursor.fetchall()[0][0]
print("Number of cases where the max excitatory firing rate is higher than 100.0 and speed=1.0 :",
      total_number)
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE speed == 1.0'
)
print("Number of cases where speed=1.0 :", cursor.fetchall()[0][0], Fore.RED + " same number than before" + Fore.RESET)

# check which cases if max is higher 100Hz for specific region region
save_result = []
for metric in range(68):
    cursor = con.cursor()
    cursor.execute(
        'SELECT ' + column + ' FROM exploration WHERE max_FR_e_' + str(metric) + '=' +
        ' (SELECT MAX(max_FR_e_' + str(metric) + ')' +
        ' FROM exploration WHERE speed == 1.0) AND max_FR_e_' + str(metric) + '>100.0'
    )
    data_ftech = cursor.fetchall()
    if len(data_ftech) > 0:
        save_result.append([metric, -1, data_ftech[0][metric], -1])
        cursor.execute(
            'SELECT ' + column + ' FROM exploration WHERE speed == 1.0 AND max_FR_e_' + str(metric) + '>100.0'
        )
        data_ftech = cursor.fetchall()
        save_result[-1][1] = np.unique(np.where(np.array(data_ftech) > 100.0)[1])
        cursor.execute(
            'SELECT COUNT() FROM exploration WHERE speed == 1.0 AND max_FR_e_' + str(metric) + '>100.0'
        )
        data_ftech = cursor.fetchall()
        save_result[-1][3] = data_ftech[0][0]

regions_high_speed = []
for region, all_region, max_firing, number in save_result:
    print("Region which is higher than 100Hz: ", region,
          "region higher than 100Hz in same time: ", all_region,
          "maximum of firing rate: ", max_firing,
          "number of cases: ", number,
          "/", total_number)
    regions_high_speed.append(region)
print(Fore.RED + "Region 59 is higher in all cases " + Fore.RESET)
print(Fore.RED + "Maximum of two regions with high activities in same time" + Fore.RESET)

print(Fore.RED + "\nAnalyse region" + Fore.RESET)
print("region with the max of connection:",
      "\n axis 0 ", str(np.argsort(np.sum(connectome.weights, axis=0))[-10:]),
      "\n axis 1 :", str(np.argsort(np.sum(connectome.weights, axis=1))[-10:])
      )
for region in regions_high_speed:
    print('region connected to the region ' + str(region) + ':',
          "\n axis 0:", str(np.argsort(connectome.weights[region, :])[-10:]),
          str(np.argmax(connectome.weights[region, :])),
          "\n axis 1:", str(np.argsort(connectome.weights[:, region])[-10:]),
          str(np.argmax(connectome.weights[:, region]))
          )
region = 59
print('region with high connection with the region ' + str(region) + ': ', end=' ')
for i in range(connectome.weights.shape[0]):
    if np.argmax(connectome.weights[:, i]) == region or np.argmax(connectome.weights[i, :]) == region:
        print(i, end=' ')

# check which cases if no noise no activity (a_e !=10 => excitation)
print(Fore.RED + "\nAnalyse no activity (mean excitatory firing rate < 1Hz) with low noise" + Fore.RESET)
cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND max_FR_e>1.0'
)
count_total = cursor.fetchall()[0][0]

cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND max_FR_e > 1.0 AND a_e == -10'
)
count_a_e = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND a_e == -10'
)
count_total_a_e = cursor.fetchall()[0][0]

cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND max_FR_e > 1.0 AND speed == 1'
)
count_speed = cursor.fetchall()[0][0]

cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND speed == 1'
)
count_total_speed = cursor.fetchall()[0][0]

cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND max_FR_e > 1.0 AND speed == 1 AND a_e == -10'
)
count_join_speed_a_e = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT COUNT() FROM exploration WHERE (wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862) AND speed == 1 AND a_e == -10'
)
count_join_total_speed_a_e = cursor.fetchall()[0][0]

print('Number of total cases for low noise and firing rate higher than 1Hz: ', count_total)
print('Number of total cases where a_e =-10 :', count_a_e, '/', count_total_a_e)
print('Number of total cases where speed=1.0 :', count_speed, '/', count_total_speed)
print('Number of total case where speed=1.0 and a_e=-10 :', count_join_speed_a_e, '/', count_join_total_speed_a_e)
print('total :(', count_total, ') = join (', count_join_speed_a_e,
      ') + a_e=-10. (', count_a_e-count_join_speed_a_e,
      ') + speed=1. (', count_speed - count_join_speed_a_e, ')')
print(Fore.RED + 'For low noise, the necessary condition for having activity is speed=1. or a_e=-10' + Fore.RESET)
print(Fore.RED + 'For low noise, the sufficient condition for having activity is speed=1. and a_e=-10' + Fore.RESET)

# check high activity
print(Fore.RED + "\nAnalyse for high activity of all brain regions (mean excitatory firing rate >100Hz)" + Fore.RESET)
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_FR_e>100.0'
)
count_all = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_FR_e>100.0 and a_e=-10.0'
)
count_a_e = cursor.fetchall()[0][0]
print('Number of total case where a_e=-10 and mean excitatory firing rate > 100Hz :', count_a_e, '/', count_all)
print(Fore.RED + 'For high activity of all brain regions, the necessary condition is a_e=-10' + Fore.RESET)

# high PLI ( condition: 1152 PLI_e>0.33: 927 PLI_i>0.3: 991)
print(Fore.RED + "\nAnalyse for synchronization : mean_PLI>0.33" + Fore.RESET)
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_PLI_e>0.33'
)
count_pli_e = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_PLI_e>0.33 and ' +
    'ex_ex == 0.0005000000237487257 and a_e == -10 and tau_w_e== 250 and speed !=1 and wNoise<1.e-5'
)
count_cond_pli_e = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE ' +
    'ex_ex == 0.0005000000237487257 and a_e == -10 and tau_w_e== 250 and speed !=1 and wNoise<1.e-5'
)
count_total_cond_pli_e = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_PLI_i>0.33'
)
count_pli_i = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_PLI_i>0.33 and ' +
    'ex_ex == 0.0005000000237487257 and a_e == -10 and tau_w_e== 250 and speed !=1 and wNoise<1.e-5'
)
count_cond_pli_i = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_PLI_i>0.33 and mean_PLI_e>0.33'
)
count_cond_pli_i_e = cursor.fetchall()[0][0]

print('Number of total case where mean_pli_e>0.33 :', count_pli_e)
print('Number of total case where mean_pli_i>0.33 :', count_pli_i)
print('NUmber of total case where mean_pli_i>0.33 and mean_pli_e>0.33:', count_cond_pli_i_e)
print('Condition : ex_ex == 0.000500000023748726 and a_e == -10 and tau_w_e== 250 and speed !=1 and wNoise<1.e-5')
print('Number of total cases with this condition:', count_total_cond_pli_e)
print('Number of condition where mean_pli_e>0.33:', count_cond_pli_e, 'mean_pli_i>0.33:', count_pli_i)
print(Fore.RED + 'For mean PLI>0.33, the necessary condition is that ex_ex == 0.000500000023748726 and a_e == -10 and tau_w_e== 250 and speed !=1 and wNoise<1.e-5' + Fore.RESET)

# high mean FC ( condition: 1536 FC_e>0.98: 1152 (+5outside condition))
print(Fore.RED + "\nAnalyse for synchronization : mean_FC_e>0.98" + Fore.RESET)
cursor = con.cursor()
cursor.execute(
    'SELECT Count() FROM exploration WHERE mean_FC_e>0.98'
)
count_fc_e = cursor.fetchall()[0][0]

cursor = con.cursor()
cursor.execute(
    'SELECT count() FROM exploration WHERE wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862 and a_e == -10 and speed !=1.0 and ( (tau_w_e==250 and ex_ex>0.00039999998989514 and ex_ex<0.00039999998989516) or (tau_w_e!=250 and ex_ex>0.000500000023748725 and ex_ex<0.000500000023748727))'
    # 'SELECT count() FROM exploration WHERE ' +
    # 'wNoise==0.00000010000000116861 and a_e == -10 and speed !=1.0 and ( (tau_w_e==250 and ex_ex==0.00039999998989515) or (tau_w_e!=250 and ex_ex==0.000500000023748726))'
)
count_cond_total_fc_e = cursor.fetchall()[0][0]
cursor = con.cursor()
cursor.execute(
    'SELECT count() FROM exploration WHERE mean_FC_e>0.98 and wNoise>0.00000010000000116860 and wNoise<0.00000010000000116862 and a_e == -10 and speed !=1.0 and ( (tau_w_e==250 and ex_ex>0.00039999998989514 and ex_ex<0.00039999998989516) or (tau_w_e!=250 and ex_ex>0.000500000023748725 and ex_ex<0.000500000023748727))'
    # 'SELECT count() FROM exploration WHERE mean_FC_e>0.98 and ' +
    # 'wNoise==0.00000010000000116861 and a_e == -10 and speed !=1.0 and ( (tau_w_e==250 and ex_ex==0.00039999998989515) or (tau_w_e!=250 and ex_ex==0.000500000023748726))'
)
count_cond_fc_e = cursor.fetchall()[0][0]
print('Number of total case where mean_FC_e>0.98:', count_fc_e)
print('Condition: wNoise==0.00000010000000116861 and a_e == -10 and speed !=1.0 and ((tau_w_e==250 and ex_ex==0.00039999998989515) or (tau_w_e!=250 and ex_ex==0.000500000023748726))')
print('Number of total cases with this condition:', count_cond_total_fc_e)
print('Number of Number of condition where mean_FC_e>0.98:', count_cond_fc_e)
print(Fore.RED + 'For mean_FC_e>0.98, one closest condition is that: wNoise==0.00000010000000116861 and a_e == -10 and speed !=1.0 and ((tau_w_e==250 and ex_ex==0.00039999998989515) or (tau_w_e!=250 and ex_ex==0.000500000023748726))' + Fore.RESET)
