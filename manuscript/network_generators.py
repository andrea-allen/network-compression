import numpy as np
import networkx as nx
import random
from src.temporalnetwork import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    degree_distribution = [1/100, 18 / 100, 58 / 100, 1 / 100, 1/100, 1/100, 1/100, 10 / 100, 1/100, 1/100, 1/100, 1/100, 5 / 100]
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

def enumerate_string_nodes(df):
    named_nodes = set(list(df['i'].unique()) + list(df['j'].unique()))
    number_labels = np.arange(len(named_nodes))
    node_dict = {a : x for (a,x) in zip(named_nodes, number_labels)}
    return node_dict

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
    elif 'RFID' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=0)
        df.columns = ['t', 'i', 'j', 'datetime']
        enumerated_nodes = enumerate_string_nodes(df)
        df = df.replace({"i": enumerated_nodes})
        df = df.replace({"j": enumerated_nodes})
    elif 'High-School' in filename:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j', 'Ci', 'Cj']
    elif 'ht09' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j']
    elif 'office' in filename:
        df = pd.read_csv(filename, delimiter=' ', header=None)
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
    elif 'RFID' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=0)
        df.columns = ['t', 'i', 'j', 'datetime']
    elif 'High-School' in filename:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j', 'Ci', 'Cj']
    elif 'ht09' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j']
    elif 'office' in filename:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j']
    else:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j']
    info = {}
    unique_timestamps = set(df['t'])
    info['num_timestamps'] = len(unique_timestamps)
    info['max_timestamp'] = max(unique_timestamps)
    info['min_timestamp'] = min(unique_timestamps)
    info['total_time'] = info['max_timestamp'] - info['min_timestamp']
    info['num_snapshots'] = len(df)
    ## WANT: Distribution of contacts per timestamp
    ## Distribution of frequency of the same contact
    histo = plt.hist(df.groupby(['i', 'j']).size(), bins='auto')
    fig, ax = plt.subplots(4, 1)
    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()
    _, ax3 = plt.subplots()
    _, ax4 = plt.subplots()
    ax = [ax1, ax2, ax3, ax4]
    # fig.set_size_inches(8,4)
    ax[0].hist(df.groupby(['t']).size(), bins='auto') # this is how many contacts per timestep
    info['contacts_per_t_mean'] = np.mean(df.groupby(['t']).size())
    info['contacts_per_t_var'] = np.var(df.groupby(['t']).size())
    ax[0].semilogy()
    ax[0].set_xlabel('Contacts \n per timestamp')
    ax[0].set_ylabel('Distribution (log)')
    ax[1].scatter(np.log10(np.arange(len(histo[0]))), np.log10(histo[0]), s=8, color='blue')
    ax[1].set_xlabel('Frequency of same\n contact (log scale)')
    ax[1].set_ylabel('Distribution (log)')
    info['same_contacts_occurance_mean'] = np.mean(df.groupby(['i', 'j']).size())
    info['same_contacts_occurance_var'] = np.var(df.groupby(['i', 'j']).size())
    static_dd = df.groupby('i')['j'].nunique()
    ax[2].hist(static_dd, color='green', alpha=0.6, density='true', label=f'<k>={np.round(np.mean(static_dd), 1)}')
    # check: set(df.where(df['i']==122).dropna()['j'])
    ax[2].set_xlabel('Number of unique contacts')
    # ax[2].set_xlabel('Degree in static network')
    ax[2].set_ylabel('Distribution')
    ax[2].legend()
    info['unique_contacts_mean'] = np.mean(static_dd)
    info['unique_contacts_var'] = np.var(static_dd)
    durations = df['t'].diff(1).dropna() # really weird giant duration gap?
    ax[3].scatter(np.arange(len(durations)), durations + 1, s=8, color='green') # +1 to make log scale work, still effectively 0
    ax[3].semilogy()
    ax[3].set_xlabel('Time steps \nin dataset')
    ax[3].set_ylabel('Time between consecutive timestamps')
    info['durations_mean'] = np.mean(durations)
    info['durations_var'] = np.var(durations)
    plt.tight_layout(0.1)
    plt.show()

    table = pd.DataFrame(info.items()).to_latex()
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
        print(f'Creating snapshot {i}')
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

def display_synthetic():
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

    all_networks = [A0, A0B, A2, A3, A4, A5, A6, A0]
    colors=sns.color_palette("Paired", n_colors=8)

    show_all=True
    if show_all:
        fig, axs = plt.subplots(2, 4, tight_layout=True, sharey=True)
        fig.set_size_inches(9,5)
        next = 0
        for i in range(2):
            for j in range(4):
                # network_graph = nx.from_numpy_array(all_networks[next])
                net_hist = np.histogram(np.sum(all_networks[next], axis=1), bins=np.arange(0,200))
                axs[i, j].bar(net_hist[1][1:], net_hist[0], linewidth=1, alpha=1, color=colors[next])                # pos = nx.spring_layout(network_graph)
                # nx.draw_networkx_nodes(network_graph, pos=pos, ax=axs[i,j], node_color="black", node_size=1)
                # nx.draw_networkx_edges(network_graph, pos=pos, ax=axs[i,j], edge_color="grey", width=0.5)
                axs[i, j].text(0.9, 0.9, f'({next + 1})', horizontalalignment='center', verticalalignment='center',
                               transform=axs[i, j].transAxes)
                axs[i,j].set_xlim([0,20])
                axs[i,j].set_xlabel("degree")
                axs[i,j].set_ylabel("number nodes")
                next += 1
        axs[0,2].set_xlim([90,110])

    show_all=True
    if show_all:
        fig2, axs = plt.subplots(2, 4, tight_layout=True)
        fig2.set_size_inches(9,5)

        next = 0
        for i in range(2):
            for j in range(4):
                network_graph = nx.from_numpy_array(all_networks[next])
                pos = nx.spring_layout(network_graph)
                nx.draw_networkx_edges(network_graph, pos=pos, ax=axs[i,j], edge_color="grey", width=0.2)
                nx.draw_networkx_nodes(network_graph, pos=pos, ax=axs[i,j], node_size=1, node_color=colors[next])
                axs[i, j].text(0.1, 0.9, f'({next + 1})', horizontalalignment='center', verticalalignment='center',
                               transform=axs[i, j].transAxes)
                next += 1

    plt.show()


