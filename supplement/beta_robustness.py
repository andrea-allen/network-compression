"""
Thesis supplement experiment to show beta robustness to a point
"""

import numpy as np
import pandas as pd

from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import matplotlib.pyplot as plt
import manuscript.network_generators as netgen
import manuscript.constants as constants
import gzip
plt.rcParams

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

# 20 betas
betas = [.0005, .001, .00135, .0017, .0021, .0025] # for synthetic
betas = [.000001, .000002, .0000035, .000005, .00001, .000015, .00002, .000025, .00003, .000035, .00004, .00005] # for hospital
b_len = len(betas)
# num_snaps_left = 5 # synthetic
num_snaps_left = 6
snapshot_boundary_results = np.zeros((b_len, num_snaps_left)) # start time of each new boundary,needs to be a 2d array for each experiment
integrated_optimal_results = np.zeros(b_len)
integrated_even_results = np.zeros(b_len)

average_degree_results = np.zeros((b_len,num_snaps_left))


#### For Hospital
"""
Split data into snapshots with specified number of snapshots
"""
# Durations with 200 snapshots: 1737
num_snapshots = 200
hospital_interval = int(constants.HOSP_TOTAL_TIME / num_snapshots)
# hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
#                                                    hospital_interval, .000015,
#                                                    int(num_snapshots / 2),
#                                                    constants.HOSP_START_TIME)


#number_of_compressions
# C = 45 # synthetic
C = 194
do_experiment = False
if do_experiment:
    for i, b in enumerate(betas):
        print(b)
        # t_interval=5
        t_interval=hospital_interval
        # a_temporal_network = synthetic_demo1(t_interval=5, beta=b)
        # a_temporal_network = TemporalNetwork(hospital_snapshots)
        a_temporal_network = load_network(f'../supplement_data/hospital_tempnet_200')
        hospital_snapshots = a_temporal_network.snapshots
        for snapshot in hospital_snapshots:
            snapshot.set_new_beta(b)
            a_temporal_network = TemporalNetwork(hospital_snapshots)

        even_t, even_inf, even_net = run_even(a_temporal_network, t_interval, b, a_temporal_network.length, C)
        temp_t, temp_inf, temp_net = run_temporal(a_temporal_network, t_interval, b, a_temporal_network.length)
        opt_t, opt_inf, opt_net, tce = run_optimal(a_temporal_network, t_interval, b,
                                                   a_temporal_network.length, C,
                                                   'combined', order=1, norm=2)
        total_optimal_error_nm = integrate_error_ts(temp_t, temp_inf, opt_t,
                                                    opt_inf)  # integral of normalized distances from temporal time series
        total_even_error_nm = integrate_error_ts(temp_t, temp_inf, even_t, even_inf)
        start_times = np.array([snap.start_time for snap in opt_net.snapshots])
        avg_degrees = np.array([np.sum(snap.dd_normalized * np.arange(len(snap.dd_normalized))) for snap in opt_net.snapshots])

        snapshot_boundary_results[i] = start_times
        average_degree_results[i] = avg_degrees
        integrated_optimal_results[i] = total_optimal_error_nm
        integrated_even_results[i] = total_even_error_nm

        results_table = pd.DataFrame(np.array([np.array(start_times), np.array(avg_degrees),
                                                  np.full(6, total_optimal_error_nm), np.full(6, total_even_error_nm)]).T,
                                        columns=["stimes", "k", "opt error", "even error"])
        results_table.to_csv(f"../supplement_data/hospital_beta_results_{b}.csv")

# plt.plot(betas, integrated_optimal_results)
# plt.plot(betas, integrated_even_results)
# plt.show()
#
# palette = sns.color_palette("Set2", n_colors=len(betas))
# for i, b in enumerate(betas):
#     plt.vlines(snapshot_boundary_results[i], ymin=i, ymax=i+1, color=palette[i], label=f'{b}')
# plt.xlabel('Snapshot times')
# plt.legend()
# plt.show()
#
# for i, b in enumerate(betas):
#     plt.scatter(average_degree_results[i],np.full(6, b), color=palette[i], label=f'{b}')
# plt.xlabel('Snapshot Average Degree')
# plt.ylabel('Beta value')
# plt.legend()
# plt.show()

sns.set_palette("Set1")
results = {}
for b in betas:
    results[b] = pd.read_csv(f"../supplement_data/hospital_beta_results_{b}.csv")


optimal_error_per_beta = np.array([v['opt error'][0] for _, v in results.items()])
even_error_per_beta = np.array([v['even error'][0] for _, v in results.items()])
plt.scatter(betas, optimal_error_per_beta, label='Algorithm')
plt.plot(betas, optimal_error_per_beta)
plt.scatter(betas, even_error_per_beta, label='Even splits')
plt.plot(betas, even_error_per_beta)
plt.xlabel('$\\beta$ value')
plt.xticks(betas, betas, rotation=-45)
plt.ylabel('Integrated error')
plt.legend()
plt.show()

palette = sns.color_palette("Blues", n_colors=len(betas))
a_temporal_network = load_network(f'../supplement_data/hospital_tempnet_200')
b = max(betas)
temp_t, temp_inf, temp_net = run_temporal(a_temporal_network, hospital_interval, b, a_temporal_network.length)
max_inf = max(temp_inf)
for i, b in enumerate(betas):
    a_temporal_network = load_network(f'../supplement_data/hospital_tempnet_200')
    hospital_snapshots = a_temporal_network.snapshots
    for snapshot in hospital_snapshots:
        snapshot.set_new_beta(b)
        a_temporal_network = TemporalNetwork(hospital_snapshots)
    temp_t, temp_inf, temp_net = run_temporal(a_temporal_network, hospital_interval, b, a_temporal_network.length)
    if i == len(betas)-1:
        plt.plot(temp_t, temp_inf, color=palette[i], lw=2, ls='--',
                 label='Fully temporal solution $I(t)$\nwith infection rate $\\beta$')
        plt.vlines(results[b]['stimes'], ymin=(i * max_inf) / 10, ymax=((i + 1) * max_inf) / 10, color=palette[i], lw=6,
                   label='Boundaries of compressed\nsnapshots')
    else:
        plt.plot(temp_t, temp_inf, color=palette[i],lw=2, ls='--')
        plt.vlines(results[b]['stimes'], ymin=(i * max_inf) / 10, ymax=((i + 1) * max_inf) / 10, color=palette[i], lw=6)
plt.xlabel('Time')
plt.ylabel('$\\beta$ value')
plt.yticks(np.arange(len(betas))*max_inf/10, betas)
plt.legend(loc='lower right')
plt.show()

print(results)

### Good for now (Saturday 8/6) do same thing for pre-compression (with hospital data)
# actually do this w hospital data w a different pre-compression level and compare

