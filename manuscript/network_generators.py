import numpy as np
import networkx as nx
import random
from src.temporalnetwork import *
import matplotlib.pyplot as plt
import pandas as pd

"""
Custom network generator functions for use in example code
Includes generalized synthetic networks and networks from real data
"""


def erdos_renyi_graph(N, p):
    G1 = nx.generators.erdos_renyi_graph(n=N, p=p)
    G1.add_nodes_from(np.arange(N))
    adj_1 = np.array(nx.adjacency_matrix(G1).todense())
    # nx.draw(G1)
    # plt.show()
    return G1, adj_1


def configuration_model_graph(N):
    degree_distribution = [0, 20 / 100, 65 / 100, 0 / 100, 0, 0, 0, 10 / 100, 0, 0, 0, 0, 5 / 100]
    got_config_model = False
    while not got_config_model:
        try:
            config_model = nx.generators.configuration_model(
                np.random.choice(np.arange(len(degree_distribution)), p=degree_distribution, size=N))
            got_config_model = True
        except:
            got_config_model = False
    config_model.add_nodes_from(list(np.arange(N)))
    config_adj = np.array(nx.adjacency_matrix(config_model).todense())
    return config_model, config_adj


def sbm(N, groups, probs):
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    node_groups = {g: [] for g in range(groups)}
    nodes_per_group = int(N / groups)
    blocks = []
    for g in range(groups):
        for n in range(g * nodes_per_group, (g + 1) * nodes_per_group):
            node_groups[g].append(n)
        # H = nx.Graph()
        # H.add_nodes_from(node_groups[g])
        H = nx.generators.erdos_renyi_graph(nodes_per_group, probs[g])
        H_nodes = list(H.nodes())
        label_map = {H_nodes[i]: node_groups[g][i] for i in range(nodes_per_group)}
        H = nx.relabel_nodes(H, label_map)
        blocks.append(H)
    for block in blocks:
        # G.add_nodes_from(block.nodes())
        G.add_edges_from(block.edges())
    for g in range(groups):
        for j in range(g, groups):
            try:
                p = probs[(g, j)]
                for n in node_groups[g]:
                    for m in node_groups[j]:
                        flip_coin = random.random()
                        if flip_coin < p:
                            G.add_edge(n, m)
            except KeyError:
                print(g, j)
                pass
    adj = np.array(nx.adjacency_matrix(G).todense())
    return G, adj


def cycle_graph(N):
    G3 = nx.generators.cycle_graph(n=N)
    G3.add_nodes_from(np.arange(N))
    Cycle_adj = np.array(nx.adjacency_matrix(G3).todense())
    return G3, Cycle_adj


def barbell_graph(N):
    G = nx.generators.barbell_graph(int(N / 2), 0)
    adj = np.array(nx.adjacency_matrix(G).todense())
    return G, adj

def produce_snapshots(df, t_start, increment):
    snapshot_1 = df[['i', 'j']].where((df['t'] >= t_start) & (df['t'] < t_start+increment))
    snapshot_2 = df[['i', 'j']].where((df['t'] >= t_start+increment) & (df['t'] < t_start+2*increment))
    return snapshot_1, snapshot_2

def graphOf(snapshot, second_snapshot):
    """
    Makes a networkx graph of nodes AND EDGES from snapshot1, (first arugment)
    Adds nodes from second_snapshot to preserve same dimensions
    :param snapshot:
    :param second_snapshot:
    :return:
    """
    snapshot = snapshot.dropna()
    snapshot['i'] = snapshot['i'].astype(int)
    snapshot['j'] = snapshot['j'].astype(int)
    second_snapshot = second_snapshot.dropna()
    second_snapshot['i'] = second_snapshot['i'].astype(int)
    second_snapshot['j'] = second_snapshot['j'].astype(int)
    graph = nx.from_pandas_edgelist(snapshot.dropna(), 'i', 'j')
    second_snapshot_graph = nx.from_pandas_edgelist(second_snapshot.dropna(), 'i', 'j')
    graph.add_nodes_from(nx.nodes(second_snapshot_graph))
    # nx.draw(graph)
    # plt.show()
    return graph


def parse_data(filename, t_start, increment):
    if 'Hospital' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j', 'i_type', 'j_type']
        df = df[['t', 'i', 'j']]
    elif 'listcontacts' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j']
    else:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j']
    print(f'There are {len(set(df["i"]).union(set(df["j"])))} nodes total')
    snapshot1, snapshot2 = produce_snapshots(df, t_start, increment)
    return graphOf(snapshot1, snapshot2), graphOf(snapshot2, snapshot1)

def dataset_statistics(filename):
    """
    Compute some statistics on the given temporal network
    :param filename:
    :return:
    """
    if 'Hospital' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j', 'type_i', 'type_j']
    elif 'listcontacts' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j']
    else:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j']
    info = {}
    unique_timestamps = set(df['t'])
    info['num_timestamps'] = len(unique_timestamps)
    info['max_timestamp'] = max(unique_timestamps)
    info['min_timestamp'] = min(unique_timestamps)
    ## WANT: Distribution of contacts per timestamp
    ## Distribution of frequency of the same contact
    histo = plt.hist(df.groupby(['i', 'j']).size(), bins='auto')
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(8,4)
    ax[0].hist(df.groupby(['t']).size(), bins='auto')  # this is how many contacts per timestep
    ax[0].semilogy()
    ax[0].set_xlabel('Contacts (edges)\n per timestamp')
    ax[0].set_ylabel('Distribution (log)')
    ax[1].scatter(np.log10(np.arange(len(histo[0]))), np.log10(histo[0]), s=8, color='blue')
    ax[1].set_xlabel('Frequency of same\n contact (log scale)')
    ax[1].set_ylabel('Distribution (log)')
    static_dd = df.groupby('i')['j'].nunique()
    ax[2].hist(static_dd, color='green', alpha=0.6, density='true', label=f'<k>={np.round(np.mean(static_dd), 1)}')
    # check: set(df.where(df['i']==122).dropna()['j'])
    ax[2].set_xlabel('Degree in static network')
    ax[2].set_ylabel('Distribution')
    ax[2].legend()
    durations = df['t'].diff(1).dropna() # really weird giant duration gap?
    ax[3].scatter(np.arange(len(durations)), durations + 1, s=8, color='green') # +1 to make log scale work, still effectively 0
    ax[3].semilogy()
    ax[3].set_xlabel('Time steps \nin dataset')
    ax[3].set_ylabel('Time between consecutive timestamps')
    plt.tight_layout(0.1)
    plt.show()

    return info


def data_network_snapshots(filename, interval, beta, num_snapshots, start_t):
    graphs = []
    start_times = []
    end_times = []
    min_real_t = start_t
    for i in range(num_snapshots):
        data = parse_data(filename, start_t, interval)
        graphs.append(data[0])
        print(f'number of nodes in snapshot: {len(data[0].nodes())}')
        start_times.append(start_t - min_real_t)
        end_times.append(start_t+interval - min_real_t)
        graphs.append(data[1])
        start_times.append(start_t+interval - min_real_t)
        end_times.append(start_t+2*interval - min_real_t)
        start_t += 2*interval
    print(len(graphs))
    all_nodes = list(graphs[0].nodes())
    for graph in graphs:
        all_nodes.extend(list(graph.nodes()))
    all_nodes = set(all_nodes)
    new_labels = {sorted(all_nodes)[i]:i for i in range(len(all_nodes))}
    relabeled_graphs = []
    for graph in graphs:
        relabeled_graphs.append(nx.relabel_nodes(graph, new_labels))
    adj_m = []
    for graph in relabeled_graphs:
        matrix = np.zeros((len(all_nodes), len(all_nodes)))
        for edge in graph.edges():
            matrix[edge[0], edge[1]] = 1
            matrix[edge[1], edge[0]] = 1
        check = np.sum(matrix - matrix.T)
        if check != 0:
            print(f'failed check on graph with sum {check}')
        adj_m.append(matrix)
    final_snapshots = []
    for i, A in enumerate(adj_m):
        final_snapshots.append(Snapshot(start_time=start_times[i], end_time=end_times[i], beta=beta, A=A))
    return final_snapshots

def synthetic_demo1(t_interval, beta):
    ### Creating the networks:
    N = 200
    A0 = np.zeros((N, N))
    A0[50,21] = 1
    A0[21,50] = 1
    A0[27,42] = 1
    A0[42,27] = 1
    A0B = np.zeros((N, N))
    A0B[57,49] = 1
    A0B[49,57] = 1
    A0B[13,4] = 1
    A0B[4,13] = 1
    G2, A2 = barbell_graph(N)
    G3, A3 = configuration_model_graph(N)
    G4, A4 = erdos_renyi_graph(N, .03)
    G5, A5 = configuration_model_graph(N)
    G6, A6 = erdos_renyi_graph(N, .01)

    ordered_pairs = [A2, A2, A0, A0B, A0,A0B, A0B, A0, A0, A0,A0, A0, A0, A0, A2,A2, A3, A4, A0, A0B,A0, A0B, A2, A5, A6,
                     A2, A0, A2, A4, A5,A3, A3, A4, A4, A0, A0, A0,A0, A4, A5,A3, A3, A4, A4, A5,A0, A0, A4, A4, A5,]

    snapshots = []
    current_start_t = 0
    for l in range(len(ordered_pairs)):
        snapshots.append(
            Snapshot(current_start_t, t_interval + current_start_t, beta, ordered_pairs[l]))
        current_start_t += t_interval
    return TemporalNetwork(snapshots)