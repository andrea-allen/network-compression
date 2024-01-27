# edgesets: list of sets. the s-th set contains all the edges (i,j) in the s-th network in the sample
# (do not include the other direction (j,i) if network is undirected). the order of edgesets within D only matters
# for contiguous clustering, where we want the edgesets to be in order of the samples in time
import mdl_functions as mdl
import simulation_helpers
import src.temporalnetwork as tn
import numpy as np
import plotters as pl
import src.compression as cs
import matplotlib.pyplot as plt


def make_edgesets(network):
    # list of sets
    set_list = []
    # provide a set for each network of edges (i,j)
    for s in range(len(network)):
        snapshot_s = network[s]
        set_s = []
        for i in range(len(snapshot_s.A)):
            for j in range(0, i):
                if snapshot_s.A[i][j] == 1:
                    set_s.append((i, j))
        set_list.append(set(set_s))
    return set_list


def make_adjacency(edge_set, N):
    edge_list = list(edge_set)
    A = np.zeros((N,N))
    for edge in edge_list:
        i, j = edge
        A[i][j] = 1
        A[j][i] = 1
    return A


def make_network_from_solution(mdl_solution, original):
    # list of edgesets
    index_list = mdl_solution["C"]
    edge_list = mdl_solution["A"]
    num_snapshots = len(index_list)
    n = original[0].N
    new_snapshot_network = []
    # First need to order the solution dict into an ordered list
    min_snapshot_keys = {min(index_list[i]): i for i in index_list.keys()}
    key_order = sorted(list(min_snapshot_keys.keys()))
    # need it to be of form min number : index, then order the min numbers, then have order of the index
    for i in range(num_snapshots):
        index_number = min_snapshot_keys[key_order[i]]
        network_list_i = index_list[index_number]
        S = tn.Snapshot(start_time=original[min(network_list_i)].start_time,
                        end_time=original[max(network_list_i)].end_time,
                        beta=beta,
                        A=make_adjacency(edge_list[index_number], N=n))
        new_snapshot_network.append(S)
    return new_snapshot_network

def aggregate_network_from_modes(mdl_solution, original):
    # list of edgesets
    index_list = mdl_solution["C"]
    edge_list = mdl_solution["A"]
    num_snapshots = len(index_list)
    n = original[0].N
    new_snapshot_network = []
    # First need to order the solution dict into an ordered list
    min_snapshot_keys = {min(index_list[i]): i for i in index_list.keys()}
    key_order = sorted(list(min_snapshot_keys.keys()))
    # need it to be of form min number : index, then order the min numbers, then have order of the index
    for i in range(num_snapshots):
        index_number = min_snapshot_keys[key_order[i]]
        network_list_i = sorted(index_list[index_number])
        new_snapshot = original[network_list_i[0]]
        Compressor = cs.Compressor()
        for s in range(1, len(network_list_i)):
            new_snapshot = Compressor.aggregate(new_snapshot, original[network_list_i[s]])
        new_snapshot_network.append(new_snapshot)
    return new_snapshot_network


import network_generators as netgen
# hs_metadata = netgen.dataset_statistics('../raw_data/High-School_data_2013.csv')
# # plt.show()
# num_snapshots = 80
# # num_snapshots = 1000 # use .000065
# # num_snapshots = 4000 # use .0002
# beta = .000015
# # beta = .000025
# snapshots = netgen.data_network_snapshots('../raw_data/High-School_data_2013.csv',
#                                                      int(hs_metadata['total_time'] / num_snapshots), beta,
#                                                      int(num_snapshots/2),
#                                                      hs_metadata['min_timestamp'])

## [DONE] TODO: Do the optimal compression for the hospital dataset, to add a star to the comparison


def run_mdl(file_name, beta, save_file_as):
    metadata = netgen.dataset_statistics(file_name)
    num_snapshots = 200
    snapshots = netgen.data_network_snapshots(file_name,
                                                         int(metadata['total_time'] / num_snapshots), beta,
                                                         int(num_snapshots/2),
                                                         metadata['min_timestamp'])

    edgeset_list = make_edgesets(snapshots)

    print(len(edgeset_list))

    # Alternatively, just compress to fewer snapshots to start

    #For contiguous clustering, use:
    MDLobj = mdl.MDL_populations(edgeset_list,N=snapshots[0].N,K0=1,n_fails=100)
    # Ok, noticing that it's really gets slower after 50 snapshots (holding on to too much? not efficient alg?)
    ## trying on a smaller subset of the network
    ## it works on small test sets of the network, could now run it on a pre-aggregated version (maybe 50 layers)
    C,A,L = MDLobj.dynamic_contiguous()

    mdl_compression = aggregate_network_from_modes({"C":C, "A":A}, snapshots)
    mdl_temporal_network = tn.TemporalNetwork(mdl_compression)

    ## Compress the network down to the number of snapshots produced by the MDL algorithm to then match it
    original_network = tn.TemporalNetwork(snapshots)
    opt_t, opt_inf, opt_net, tce = simulation_helpers.run_optimal(original_network, int(metadata['total_time'] / num_snapshots),
                                                                  beta,
                                                       original_network.length, original_network.length - mdl_temporal_network.length + 4,
                                                       'combined', order=1, norm=2)

    ## Now perform the solution on each one, opt net vs mdl
    comparison_results = simulation_helpers.compare_mdl(opt_net, mdl_temporal_network, original_network,
                    original_network.snapshots[0].duration, beta, original_network.length - mdl_temporal_network.length)

    ## Issue here now is that the "mode" networks are all 1s or 0s, whereas our networks are now weighted as they get compressed.
    ## Should we compare against the temporal run now, our version but all 1s, against this version? Or just see where the boundaries end up?
    ## Need to make sure that in the paper, we say we end up with compressed and weighted networks
    ## I guess the solution is, we use this thing to find the mode networks, and then we just force compression at those boundaries
    # by using the same aggregation function but weighting and doing all that. Then we can compare.
    ## Use the result to instead of creating a network, just indentify the time boundaries. Then use our functions to say, ok,
    ## these are the snapshot index boundaries, aggregate all these networks and then see how things go
    pl.plot_one_round_compare_mdl(comparison_results)
    plt.show()

    ## Now get the integral of the error for the one round

    temp_t, temp_inf, temp_net = simulation_helpers.run_temporal(original_network, int(metadata['total_time'] / num_snapshots), beta, original_network.length)
    mdl_t, mdl_inf, mdl_net = simulation_helpers.run_temporal(mdl_temporal_network, original_network.snapshots[0].duration, beta, mdl_temporal_network.length)
    total_mdl_error_nm = simulation_helpers.integrate_error_ts(temp_t, temp_inf, mdl_t,
                                                mdl_inf)  # integral of normalized distances from temporal time series

    print(total_mdl_error_nm)
    print(mdl_temporal_network.length)

    mdl_length_and_error = np.array([total_mdl_error_nm, mdl_temporal_network.length])

    np.savetxt(f'results/revisions1/{save_file_as}.txt', mdl_length_and_error, delimiter=',')
    ## 8073

    print('done')

## [DONE] TODO Make a figure that has a star where the optimal compression is, where the error for it is, as a function
## of number of compressions. Then we can show how many more compressions you can do. Should probably do this for each empirical set...

## TODO NEXT: For each data set, run the MDL, output the error, save it as a text file, then plot it as an X-Y plot of
## X: optimal compression number of snapshots, Y: how many more snapshots the algorithm can compress than it

file_name = '../raw_data/detailed_list_of_contacts_Hospital.dat_'
beta = .000015
# run_mdl(file_name, beta, 'hospital')

file_name = '../raw_data/tij_InVS15_office.dat_'
beta = .00001
# run_mdl(file_name, beta, 'office')

file_name = '../raw_data/ht09_contact_list.dat'
beta = .00001
# run_mdl(file_name, beta, 'ht09')

file_name = '../raw_data/High-School_data_2013.csv'
beta = .000015
# run_mdl(file_name, beta, 'highschool')

file_name = '../raw_data/tij_InVS.dat'
beta = .000025
# run_mdl(file_name, beta, 'conference')

def compute_better_than_measure(results, mdl_file_name):
    mdl_results = np.loadtxt(f'results/revisions1/{mdl_file_name}.txt')
    mdl_layers = int(mdl_results[1])
    mdl_error = mdl_results[0]
    iter_range = results["iter_range"] # 1 2 3 4 5
    optimal_errors_norm = results['opt_error_norm']
    even_errors_norm = results['even_error_norm']
    mdl_count = 0
    for i in range(100-mdl_layers, 100):
        # mdl_error = 12542
        Alg_x = optimal_errors_norm[i]
        if Alg_x < mdl_error:
            mdl_count += 1
    factor = (200 - iter_range[100 - 23]) / (200 - iter_range[100 - 23] - mdl_count)
    return (mdl_layers, mdl_count, factor)

result_array = np.loadtxt('./results/draft2/hospital_error_integrals_updated.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
hosp_layers, hos_count, hosp_factor = compute_better_than_measure(results, 'hospital')
print(hosp_layers, hos_count, hosp_factor)

result_array = np.loadtxt('./results/revisions1/highschool_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
hs_layers, hs_count, hs_factor = compute_better_than_measure(results, 'highschool')

result_array = np.loadtxt('./results/revisions1/conf_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
conf_layers, conf_count, conf_factor = compute_better_than_measure(results, 'conference')

result_array = np.loadtxt('./results/revisions1/ht09_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
ht09_layers, ht09_count, ht09_factor = compute_better_than_measure(results, 'ht09')

result_array = np.loadtxt('./results/revisions1/office_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
office_layers, office_count, office_factor = compute_better_than_measure(results, 'office')

layers = np.array([hosp_layers, hs_layers, conf_layers, ht09_layers, office_layers])
error_count = np.array([hos_count, hs_count, conf_count, ht09_count, office_count])
compression_factor = np.array([hosp_factor, hs_factor, conf_factor, ht09_factor, office_factor])
labels = ['hospital', 'high school', 'conference', 'ht09', 'office']

mdl_ensemble_results = np.array([layers, compression_factor, error_count])
np.savetxt('./results/revisions1/mdl_data_results.txt', mdl_ensemble_results, delimiter=",")
plt.scatter(layers, error_count, label=labels, marker="*", s=60)
plt.scatter(layers, compression_factor, label=labels, marker="*", s=60)
plt.xlabel('Optimal number of network snapshots (MDL)')
plt.ylabel('Number of more layers algorithm can compress while maintaining less error')
plt.show()



## X: mdl compression number of snapshots, Y: how many more snapshots the algorithm can compress than it

#### INSET
## Plotting how many more compressions you can do with our algorithm than the MDL
# axins.scatter(200-23, (200-iter_range[100-23])/(200-iter_range[100-23]-mdl_count), s=4,
#               color="red")

