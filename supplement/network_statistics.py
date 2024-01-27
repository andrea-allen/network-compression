# Supplementary material for supplement
# Network statistics for the real and synthetic data used in the original paper, along with the
# If I'm getting a degree in data science, I feel like the committee might ask for
# information on the dataset/sets used for the validation of the paper
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import manuscript.network_generators as netgen
import manuscript.constants
import seaborn as sns
import networkx as nx
import src.temporalnetwork as tnt

# Temporal contact distribution
# First get data on the data itself, summarized in tables in the supplement
# network_generators.dataset_statistics('../raw_data/tij_InVS.dat')
# info = netgen.dataset_statistics('../raw_data/detailed_list_of_contacts_Hospital.dat_')
# Get this data as a table too
# plt.show()

# ## Then, do network statistics on the layers created into 200 snapshots, arguing that it's
# # somewhat arbitrary however much you pre-compress
# num_snapshots = 200
# snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
#                                           int(constants.HOSP_TOTAL_TIME / num_snapshots), .000015,
#                                           int(num_snapshots / 2),
#                                           constants.HOSP_START_TIME)
# info = {}
# # mean and variance number of edges per snapshot
# edge_counts = [np.sum(snap.A) / 2 for snap in snapshots]
# mean_edges = np.mean(edge_counts)
# std_edges = np.sqrt(np.var(edge_counts))
# # edge_distribution = plt.hist(edge_counts, color='red', alpha=0.5)
# # plt.vlines(mean_edges, ymin=0, ymax=max(edge_counts), label='mean')
# # plt.legend()
# # plt.show()
#
# # Clustering
# # average shortest path length
# network = nx.from_numpy_matrix(snapshots[5].A)
# # Average degree per snapshot (plot?)
# avg_degrees = np.array([np.mean([np.sum(snap.A[i]) for i in range(len(snap.A))]) for snap in snapshots])
# std_degrees = np.array([np.sqrt(np.var([np.sum(snap.A[i]) for i in range(len(snap.A))])) for snap in snapshots])
# print(avg_degrees)
# # TODO: could do this scatter plot of average degree for some different pre-compress options
# plt.scatter(np.arange(1, 201), avg_degrees, marker='o', s=10)


# can also plot + the std deviation, but, since the distribution isn't normal a violin
# plot-like thing might be better
# Average density of snapshots
#


# So, can't really do network statistics on the pre-compressed data,
# since by nature it's just literally raw data, we are generating networks on it
# BUT it would be cool to do some network statistics on the data at the end
# average shortest path length, clustering, degree distirbution of each snapshot, etc.

# Should potentially show something about the pre-compression data
# or demonstrate the results from doing pre-aggregating levels at even increments
# HAd a way to show that in the past, should remember how we did that


# # Combo idea
# beta = .1
# t = g/qb, g=qb*t
# t: 10, 15, 22, 30
# q: 2,  3,  2.5, 1.5
# g1 = 2, g2 = 3, g3 = 5.5, q4 = 4.5 (but less than before)
# that's because it's assuming qb was even for all the generations
# leading up to this one, so have to somehow incorporate the fact that the
# networks leading up to it had a different q... average all the q's?
# q_avg: 2, 2.5, 2.5, 2.25
# maybe do weighted average? by temporal time? then re-compute. but already this is good
# idea to run by LHD
# then:
# g1 = 2. g2 = 3.75, g3 = 5.5, g4 = 6.75
# rounded: 2, 4, 6, 7
# so then use net1 for 1,2.. net2 for 3,4, net3 for 5, 6, net4 for 7, onward
# Then what I would do is round off these generational numbers, and then use each
# of the empirical networks to generate the windows of distributions for
# the cumulative cases, plot those and hope they match up with the
# simulations.

def get_network_stats(temporal_network):
    """
    Method to return a table of network stats specifically for a temporal network object
    :param temporal_network:
    :return: dataframe
    """
    # Snapshot number
    snap_numbers = [i for i in range(temporal_network.length)]
    # Number nodes N
    num_nodes = [snap.N for snap in temporal_network.snapshots]
    # Average degree
    # k = [np.sum(snap.dd_normalized * np.arange(len(snap.dd_normalized))) for snap in temporal_network.snapshots]
    k = [np.mean(np.sum(snap.A, axis=0)) for snap in temporal_network.snapshots]
    # .2x^0, .3x^1, .25x^2, .5x^3
    # Average excess degree
    # (k-1)kp_k/sum kp_k
    # q = [np.sum(snap.dd_normalized[1:] * np.arange(len(snap.dd_normalized) - 1)) for snap in temporal_network.snapshots]
    # q = [max (0,np.sum([snap.dd_normalized[k]*k*(k-1) for
    #                        k in range(len(snap.dd_normalized))])/(np.sum([snap.dd_normalized[k]*k
    #                                                                       for k in range(len(snap.dd_normalized))]))) for snap in temporal_network.snapshots]
    # q_raw = np.array([np.mean([max((np.sum(snap.A, axis=0) - 1)[i],0) for i in range(75)]) for snap in temporal_network.snapshots])/np.array(k)
    # q_raw = np.array([np.array([np.array(nx.degree_histogram(nx.from_numpy_matrix(snap.A)))[i]*i*(i-1) for
    #     i in range(len(np.array(nx.degree_histogram(nx.from_numpy_matrix(snap.A)))))]) for snap in temporal_network.snapshots])/np.array(k)
    #
    q_empty = np.zeros(temporal_network.length)
    s = 0
    for snap in temporal_network.snapshots:
        dd = np.array(nx.degree_histogram(nx.from_numpy_matrix(snap.A)))/snap.N
        # q_excess = 0
        # for i in range(len(dd)):
        #     q_excess += dd[i]*i*(i-1)
        q_empty[s] = np.sum(np.arange(len(dd)-1)*np.arange(len(dd))[1:]*dd[1:])/(np.sum(dd*np.arange(len(dd))))
        s += 1

    q = q_empty

    # q = [max(0, q_raw[i]) for i in range(len(q_raw))]
    # Assortativity
    # assort = [nx.degree_assortativity_coefficient(nx.from_numpy_matrix(snap.A)) for snap in temporal_network.snapshots]
    # Clustering
    clustered = [nx.average_clustering(nx.from_numpy_matrix(snap.A)) for snap in temporal_network.snapshots]

    n_connected_components = [nx.number_connected_components(nx.from_numpy_matrix(snap.A)) for snap in temporal_network.snapshots]
    # gcc = list(max(nx.connected_components(nx.from_numpy_matrix(temporal_network.snapshots[100].A)), key=len))[0]
    prop_nodes_in_gcc = [len(max((nx.connected_components(nx.from_numpy_matrix(snap.A))), key=len)) / snap.N for snap in temporal_network.snapshots]

    stat_table = pd.DataFrame(np.array([snap_numbers, num_nodes, k, q, clustered, n_connected_components, prop_nodes_in_gcc]).T,
                              columns=['Snapshot number', 'N', 'k', 'q',
                                       'C', 'N connected components', 'prop in gcc'])

    return stat_table
