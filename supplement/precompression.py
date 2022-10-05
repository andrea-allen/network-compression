"""
Thesis supplement experiment to show beta robustness to a point
"""
import pandas as pd

from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import matplotlib.pyplot as plt
import manuscript.network_generators as netgen
import manuscript.constants as constants
import network_statistics
import gzip

## For a range of beta, do the synthetic compression (one round) and keep track of where the resulting boundaries are
## Plot the results and variance
## Keep track of the integrated error and compare that to the even error for each
## Keep track of the network stats for each resulting network and plot those

def load_network(fpath):
    with gzip.open(fpath, 'r') as fin:  # 4. gzip
        json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    loaded_network = TemporalNetworkDecoder().decode(json_str=json_str)
    return loaded_network

def save_network(fpath, network):
    encoded_network = TemporalNetworkEncoder().encode(network)
    loaded_bytes = encoded_network.encode('utf-8')
    with gzip.open(fpath, 'w') as fout:       # 4. fewer bytes (i.e. gzip)
        fout.write(loaded_bytes)


beta = .000015
precomp_level = [1000, 800, 600, 400, 300, 200, 150, 100]
precomps_len = len(precomp_level)

# ########## PRE-COMPRESSED NETWORK STATISTICS
# for i, l in enumerate(precomp_level):
#     print(l)
#     num_snapshots = l
#     hospital_interval = int(constants.HOSP_TOTAL_TIME / num_snapshots)
#     hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
#                                                        hospital_interval, .000015,
#                                                        int(num_snapshots / 2),
#                                                        constants.HOSP_START_TIME)
#     temporal_net = TemporalNetwork(hospital_snapshots)
#     # IDEALLY SAVE THE SNAPSHOTS?
#     save_network(f'hospital_tempnet_{l}', temporal_net)
#
#     t_interval=hospital_interval
#
#     # Network data
#     stats = network_statistics.get_network_stats(temporal_net)
#     stats.to_csv(f"../supplement_data/hospital_tempnet_stats_{l}")
#     sns.set_palette("Set2")
#     plt.scatter(stats['Snapshot number'], stats['C'] * 75 * 3, alpha=0.5)
#     plt.scatter(stats['Snapshot number'], stats['k'], alpha=0.5)
#     plt.scatter(stats['Snapshot number'], stats['q'], alpha=0.5)
#     plt.show()

######## PRE-COMPRESSION EXPERIMENT
# num_snaps_left = 5 # synthetic
num_snaps_left = 6
snapshot_boundary_results = np.zeros((precomps_len, num_snaps_left)) # start time of each new boundary,needs to be a 2d array for each experiment
integrated_optimal_results = np.zeros(precomps_len)
integrated_even_results = np.zeros(precomps_len)

average_degree_results = np.zeros((precomps_len,num_snaps_left))


#### For Hospital

#number_of_compressions
# C = 45 # synthetic
do_experiment = False
if do_experiment:
    for i, l in enumerate(precomp_level):
        print(l)
        num_snapshots = l
        hospital_interval = int(constants.HOSP_TOTAL_TIME / num_snapshots)
        # hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
        #                                                    hospital_interval, .000015,
        #                                                    int(num_snapshots / 2),
        #                                                    constants.HOSP_START_TIME)
        # t_interval=5
        t_interval=hospital_interval
        # a_temporal_network = synthetic_demo1(t_interval=5, beta=b)
        # a_temporal_network = TemporalNetwork(hospital_snapshots)
        a_temporal_network = load_network(f'hospital_tempnet_{l}')

        # Network data
        # stats = network_statistics.get_network_stats(a_temporal_network)
        # sns.set_palette("Set2")
        # plt.scatter(stats['Snapshot number'], stats['C'] * 75 * 3, alpha=0.5)
        # plt.scatter(stats['Snapshot number'], stats['k'], alpha=0.5)
        # plt.scatter(stats['Snapshot number'], stats['q'], alpha=0.5)
        # plt.show()

        # First do L=10, then go down to L=6, and save the results
        L = 10
        C = num_snapshots - L
        even_t, even_inf, even_net = run_even(a_temporal_network, t_interval, beta, a_temporal_network.length, C)
        temp_t, temp_inf, temp_net = run_temporal(a_temporal_network, t_interval, beta, a_temporal_network.length)
        opt_t, opt_inf, opt_net, tce = run_optimal(a_temporal_network, t_interval, beta,
                                                   a_temporal_network.length, C,
                                                   'combined', order=1, norm=2)
        print(f"Integrating the error induced by {l} snapshots compressed into {l-C}")
        total_optimal_error_nm = integrate_error_ts(temp_t, temp_inf, opt_t,
                                                    opt_inf)  # integral of normalized distances from temporal time series
        total_even_error_nm = integrate_error_ts(temp_t, temp_inf, even_t, even_inf)
        start_times = np.array([snap.start_time for snap in opt_net.snapshots])
        avg_degrees = np.array([np.sum(snap.dd_normalized * np.arange(len(snap.dd_normalized))) for snap in opt_net.snapshots])

        # snapshot_boundary_results[i] = start_times
        # average_degree_results[i] = avg_degrees
        # integrated_optimal_results[i] = total_optimal_error_nm
        # integrated_even_results[i] = total_even_error_nm
        L10_results_table = pd.DataFrame(np.array([np.array(start_times), np.array(avg_degrees),
                                          np.full(L, total_optimal_error_nm), np.full(L, total_even_error_nm)]).T,
                                         columns=["stimes", "k", "opt error", "even error"])
        L10_results_table.to_csv(f"../supplement_data/hospital_precompress_results_{num_snapshots}_to_{L}")

        L = 6
        C = num_snapshots - L
        even_t, even_inf, _ = run_even(a_temporal_network, t_interval, beta, a_temporal_network.length, C)
        opt_t, opt_inf, opt_net, _ = run_optimal(opt_net, t_interval, beta,
                                                   opt_net.length, 4, # 4 more compressions from 10 to 6
                                                   'combined', order=1, norm=2)
        print(f"Integrating the error induced by {l} snapshots compressed into {L}")
        total_optimal_error_nm = integrate_error_ts(temp_t, temp_inf, opt_t,
                                                    opt_inf)  # integral of normalized distances from temporal time series
        total_even_error_nm = integrate_error_ts(temp_t, temp_inf, even_t, even_inf)
        start_times = np.array([snap.start_time for snap in opt_net.snapshots])
        avg_degrees = np.array([np.sum(snap.dd_normalized * np.arange(len(snap.dd_normalized))) for snap in opt_net.snapshots])

        # snapshot_boundary_results[i] = start_times
        # average_degree_results[i] = avg_degrees
        # integrated_optimal_results[i] = total_optimal_error_nm
        # integrated_even_results[i] = total_even_error_nm
        L6_results_table = pd.DataFrame(np.array([np.array(start_times), np.array(avg_degrees),
                                          np.full(L, total_optimal_error_nm), np.full(L, total_even_error_nm)]).T,
                                         columns=["stimes", "k", "opt error", "even error"])
        L6_results_table.to_csv(f"../supplement_data/hospital_precompress_results_{num_snapshots}_to_{L}")

### loading network stats
network_stats = {}
for i, l in enumerate(precomp_level):
    network_stats[l] = pd.read_csv(f"../supplement_data/hospital_tempnet_stats_{l}")

### Plots of the k,q, C values for the compressed levels:
fig, ax = plt.subplots(3, 1)
sns.set_palette("Set2")
for k, v in network_stats.items():
    ax[0].hist(v['k'], density=True, alpha=0.8, bins=20, histtype=u'step')
    ax[1].hist(v['q'], density=True, alpha=0.8, bins=20, histtype=u'step')
    ax[2].hist(v['C'], density=True, alpha=0.8, bins=20, histtype=u'step')
ax[0].set_xlabel('Degree $k$')
ax[1].set_xlabel('Excess degree $q$')
ax[2].set_xlabel('Clustering coefficient $C$')
plt.tight_layout()
plt.show()

## Should also print a table of summary statistics about the mean, variance of each summary statistic, min and max


### Loading results
hc_100_6 = pd.read_csv('../supplement_data/hospital_precompress_results_100_to_6')
hc_150_6 = pd.read_csv('../supplement_data/hospital_precompress_results_150_to_6')
hc_200_6 = pd.read_csv('../supplement_data/hospital_precompress_results_200_to_6')
hc_300_6 = pd.read_csv('../supplement_data/hospital_precompress_results_300_to_6')
hc_400_6 = pd.read_csv('../supplement_data/hospital_precompress_results_400_to_6')
hc_600_6 = pd.read_csv('../supplement_data/hospital_precompress_results_600_to_6')
hc_800_6 = pd.read_csv('../supplement_data/hospital_precompress_results_800_to_6')
hc_1000_6 = pd.read_csv('../supplement_data/hospital_precompress_results_1000_to_6')

L6_results = {100: hc_100_6,150: hc_150_6,200: hc_200_6,300: hc_300_6,
              400: hc_400_6,600: hc_600_6,800: hc_800_6,1000: hc_1000_6}

hc_100_10 = pd.read_csv('../supplement_data/hospital_precompress_results_100_to_10')
hc_150_10 = pd.read_csv('../supplement_data/hospital_precompress_results_150_to_10')
hc_200_10 = pd.read_csv('../supplement_data/hospital_precompress_results_200_to_10')
hc_300_10 = pd.read_csv('../supplement_data/hospital_precompress_results_300_to_10')
hc_400_10 = pd.read_csv('../supplement_data/hospital_precompress_results_400_to_10')
hc_600_10 = pd.read_csv('../supplement_data/hospital_precompress_results_600_to_10')
hc_800_10 = pd.read_csv('../supplement_data/hospital_precompress_results_800_to_10')
hc_1000_10 = pd.read_csv('../supplement_data/hospital_precompress_results_1000_to_10')
L10_results = {100: hc_100_10,150: hc_150_10,200: hc_200_10,300: hc_300_10,
               400: hc_400_10,600: hc_600_10,800: hc_800_10,1000: hc_1000_10}

network_stats_table = pd.concat([network_stats[k].groupby(by='N').mean() for k,v in L10_results.items() ])
network_stats_table = network_stats_table[['k', 'q', 'C']]
network_stats_table['Pre-compression level'] = precomp_level[::-1]
network_stats_table = network_stats_table[['Pre-compression level', 'k', 'q', 'C']]
tex_table_network_stats = network_stats_table.to_latex()

sns.set_palette("Set1")
even_errors = [tab['even error'][0] for _, tab in L10_results.items()]
opt_errors = [tab['opt error'][0] for _, tab in L10_results.items()]
optimal_error_per_level = np.array([v['opt error'][0] for _, v in L10_results.items()])
even_error_per_level = np.array([v['even error'][0] for _, v in L10_results.items()])
plt.scatter(precomp_level, optimal_error_per_level, label='Algorithm')
plt.plot(precomp_level, optimal_error_per_level)
plt.scatter(precomp_level, even_error_per_level, label='Even splits')
plt.plot(precomp_level, even_error_per_level)
plt.xlabel('Pre-compression level')
plt.xticks(precomp_level, precomp_level, rotation=-45)
plt.ylabel('Integrated error')
plt.legend()
plt.tight_layout()
plt.show()


palette = sns.color_palette("Set2", n_colors=len(precomp_level))
palette = sns.color_palette("Blues", n_colors=len(precomp_level))
for i, l in enumerate(precomp_level):
    if i < len(precomp_level)-1:
        plt.vlines(L10_results[l]['stimes'], ymin=(i*10)/10, ymax=((i+1)*10)/10, color=palette[i], lw=6)
    else:
        plt.vlines(L10_results[l]['stimes'], ymin=(i*10)/10, ymax=((i+1)*10)/10, color=palette[i], lw=6,
                   label='Boundaries of compressed snapshots\nwith infection rate $\\beta$')
plt.xlabel('Time')
plt.ylabel('Precompression level')
plt.yticks(np.arange(len(precomp_level))*10/10, precomp_level)
plt.legend(loc='lower right')
plt.show()
