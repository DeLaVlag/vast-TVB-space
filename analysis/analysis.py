import datetime
from analysis.analyses import *

def analysis(parameter_name, parameter_value, timeseries, weights_mat, tracts_mat, dt=1.e-3, start=1000,
             ratio_threshold=0.3, len_state=20, gauss_width_ratio=10,
             psd_type_mean='avg_psd', psd_prominence=1, psd_which_peak='both',
             slope_range_fit=(0.1, 1000), slope_type_mean='avg_psd',
             bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 100)}
             ):
    time_s = np.arange(start, start + timeseries.shape[0], 1) * dt
    f_sampling = 1. / dt  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1 / f_sampling)
    FR_exc = timeseries[:, 0, :] * 1e3  # from KHz to Hz; Excitatory firing rate
    FR_inh = timeseries[:, 1, :] * 1e3  # from KHz to Hz; Inhibitory firing rate
    nb_region = FR_exc.shape[1]

    store_npy = {'date':  datetime.datetime.now(),}
    # ========================================= parameter ========================================= #
    for name, value in zip(parameter_name, parameter_value):
        store_npy[name] = float(value)

    # ========================================= FR ========================================= #
    for label, FR in [('e', FR_exc), ('i', FR_inh)]:
        # Mean and std deviation of regions
        for reg, mean_FR in zip(range(nb_region), np.mean(FR, axis=0)):
            store_npy['mean_FR_' + label + '_' + str(reg)] = float(mean_FR)
        store_npy['mean_FR_' + label] = float(np.mean(FR))
        # for reg, std_FR in zip(range(nb_region), np.std(FR, axis=0)):
        #     store_npy['std_FR_' + label + '_' + str(reg)] = float(std_FR)
        store_npy['std_FR_' + label] = float(np.std(FR))
        # Coefficient of variation
        store_npy['coeff_var_' + label] = float(np.mean(FR) / np.std(FR))
        # Maximal
        for reg, max_FR in zip(range(nb_region), np.max(FR, axis=0)):
            store_npy['max_FR_' + label + '_' + str(reg)] = float(max_FR)
        store_npy['max_FR_' + label] = float(np.amax(FR))  # We'll see here if the 200 Hz fixed point appears.
        # Minimal
        for reg, min_FR in zip(range(nb_region), np.min(FR, axis=0)):
            store_npy['min_FR_' + label + '_' + str(reg)] = float(min_FR)
        store_npy['min_FR_' + label] = float(np.amin(FR))  # We'll see here if the 200 Hz fixed point appears.

        # Average FC
        store_npy['mean_FC_' + label] = float(mean_FC(FR))

        # Average PLI
        store_npy['mean_PLI_' + label] = float(mean_PLI(FR))

        # Length of Up-Down states
        store_npy['mean_up_' + label], store_npy['mean_down_' + label] =\
            mean_UD_duration(FR, dt=dt, ratio_threshold=ratio_threshold, len_state=len_state,
                             gauss_width_ratio=gauss_width_ratio)

        # Obtaining PSDs
        psd = np.abs(np.fft.fft(FR.T)) ** 2

        # Frequency at peak and amplitude of peak of power spectra.
        f_max, amp_max = psd_fmax_ampmax(frq, psd, type_mean=psd_type_mean, prominence=psd_prominence,
                                         which_peak=psd_which_peak)
        for name, f, amp in zip(['amp_' + label, 'prom_' + label], f_max, amp_max):
            store_npy['fmax_' + name] = f
            store_npy['pmax_' + name] = amp

        for reg, freq_dom in zip(range(nb_region), frequency_analysis(recorded_signal=FR, fs=f_sampling)):
            store_npy['freq_dom_' + label + '_' + str(reg)] = freq_dom

        # Slope of power spectra
        a, b, score = fit_psd_slope(frq, psd, range_fit=slope_range_fit, type_mean=slope_type_mean)
        store_npy['slope_PSD_' + label] = a
        store_npy['score_PSD_' + label] = score  # Store the score to know the goodness of fit

        dict_rel_powers = rel_power_bands(frq, np.mean(psd, axis=0), bands, do_plot=False)

        for band in dict_rel_powers:  # Store results
            store_npy[band + '_rel_p_' + label] = dict_rel_powers[band]

        # ========================================= PREDICTIONS ========================================= #
        store_npy['ratio_frmean_dmn_' + label] = ratio_most_active_from_dmn(FR)
        FC = np.corrcoef(FR.T)
        store_npy['ratio_zscore_dmn_' + label] = ratio_zscore_from_dmn(FC)
        store_npy['ratio_AI_' + label] = count_ratio_AI(FR)  # Still under construction
        store_npy['varFCD_' + label] = varFCD(FR, FR.shape[0] - start)

        # Correlations between FC matrix and SC matrix
        store_npy['corr_FC_SC_' + label] = np.corrcoef(x=FC.flatten(), y=weights_mat.flatten())[0, 1]
        # Correlations between FC matrix and tract lengths matrix.
        # Reason: longer tract, more time to reach effect, less synchronized.
        store_npy['corr_FC_tract_' + label] = np.corrcoef(x=FC.flatten(), y=tracts_mat.flatten())[0, 1]
    return store_npy



if '__main__' == __name__:
    from tvb.simulator.lab import connectivity

    path = "/home/kusch/Documents/project/Zerlaut/compare_zerlaut/test_analysis/data/"
    parameter_name = ['g', 'be', 'wNoise', 'T', 'speed', 'tau_w_e', 'a_e', 'ex_ex', 'ex_in', 'in_ex', 'in_in']
    # TODO find a way to load them
    parameter_value = [1.00e-01, 0.00e+00, 1.00e+00, 3.40e+01, 1.00e+00, 7.50e+02, 2.00e+01, 3.00e-04, 5.00e-04,
                       5.00e-04, 3.00e-04]
    connectivity = connectivity.Connectivity().from_file(path + "connectivity_zerlaut_68.zip")
    weights_mat = np.array(connectivity.weights)
    tracts_mat = np.array(connectivity.tract_lengths)
    rank = 0
    timeseries = np.load(path + "tavg_data" + str(rank), allow_pickle=True)
    result = analysis(parameter_name, parameter_value, timeseries, weights_mat, tracts_mat)

    print(result.keys())

    # # example of result
    # example_metrics = {  # parameters
    #     'g': 1.00e-01, 'be': 0.00e+00, 'wNoise': 1.00e+00, 'T': 3.400, 'speed': 1.00e+00, 'tau_w_e': 7.50e+02,
    #     'a_e': 2.00e+01, 'ex_ex': 3.00e-04, 'ex_in': 5.00e-04, 'in_ex': 5.00e-04, 'in_in': 3.00e-04,
    #     'mean_FR_e': 5, 'mean_FR_i': 23, 'std_FR_e': 6, 'std_FR_i': 24,
    #     # Mean of FC and PLI matrices for both FRs
    #     'mean_FC_e': 7, 'mean_FC_i': 25, 'mean_PLI_e': 8, 'mean_PLI_i': 26,
    #     # Mean duration of up and down states
    #     'mean_up_e': 9, 'mean_up_i': 27, 'mean_down_e': 10, 'mean_down_i': 28,
    #     # all time max and minimum FR over all the regions
    #     'max_FR_e': 11, 'max_FR_i': 29,
    #     # Peaks of PSDs. We obtain the frequency (fmax) at which the peak appears and
    #     # its amplitude/power (pmax)
    #     # The peak has been calculted with two methods: peak with highest amplitude (amp)
    #     # and peak with highest prominence (prom)
    #     'fmax_amp_e': 12, 'pmax_amp_e': 13, 'fmax_amp_i': 30, 'pmax_amp_i': 31,
    #     'fmax_prom_e': 14, 'pmax_prom_e': 15, 'fmax_prom_i': 32, 'pmax_prom_i': 33,
    #     # other method for dominant frequency
    #     'freq_dom_e': 12, 'freq_dom_i': 12,
    #     # The PSD has also been fitted to a power law b/f^a. We have obtained the slope a and
    #     # the score of the fitting.
    #     'slope_PSD_e': 16, 'score_PSD_e': 17, 'slope_PSD_i': 34, 'score_PSD_i': 35,
    #     # frequencies have been divided into 5 typical bands: 'delta': (0.5, 4)Hz,
    #     # 'theta': (4, 8)Hz,  'alpha': (8, 12)Hz, 'beta': (12, 30) Hz, 'gamma': (30, 100) Hz
    #     # and the relative power in each band has been obtained by numerical integration
    #     'delta_rel_p_e': 18, 'theta_rel_p_e': 19, 'alpha_rel_p_e': 20, 'beta_rel_p_e': 21,
    #     'gamma_rel_p_e': 22, 'delta_rel_p_i': 36, 'theta_rel_p_i': 37, 'alpha_rel_p_i': 38,
    #     'beta_rel_p_i': 39, 'gamma_rel_p_i': 40,
    #     # Finally, scoring on how close the spontaneous dynamics are to the expected dynamics
    #     # of the DMN has been performed in two ways.
    #     # 1. zscore: take PCC as seed, obtain correlations with all other regions (zscores),
    #     # take: the 10 most correlated, count how many of the 10 regions belong to DMN
    #     # 2. frmean: obtain mean FR over time of each region, take the 10 with highest
    #     # mean FR, count how many of the 10 regions belong to DMN.
    #     'ratio_frmean_dmn_e': 41, 'ratio_zscore_dmn_e': 42, 'ratio_frmean_dmn_i': 43,
    #     'ratio_zscore_dmn_i': 44,
    #     # To be finished, trying to count how many AI or UD nodes are in the 68 regions
    #     'ratio_AI_e': 45, 'ratio_AI_i': 45,
    #     # Correlation between the FC matrix and the SC/weight matrix.
    #     'corr_FC_SC_e': 46, 'corr_FC_SC_i': 47,
    #     # Correlation between the FC matrix and the SC/weight matrix.
    #     'corr_FC_tract_e': 48, 'corr_FC_tract_i': 49,
    #     # Coefficient of variation (std/mean)
    #     'coeff_var_e': 50, 'coeff_var_i': 51,
    #     # standard deviation of the mean FR in time (so std of a 68 element vector)
    #     'std_of_means_e': 52, 'std_of_means_i': 53,
    #     # Mean of std vector in time (so mean of 68 std element vector)
    #     'means_of_std_e': 54, 'means_of_std_i': 55}
    # for key in result.keys():
    #     print(key, example_metrics[key])
    # for key in example_metrics.keys():
    #     print(result[key])
