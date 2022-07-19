import matplotlib.pyplot as plt
import numpy as np
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json
from collections import namedtuple
import gzip

"""
On synthetic dataset (pre-saved sample network designed in manuscript)
Testing whether the linear approximation works as well as the O(3) approximation
Testing whether using the operator norm works as well as the dot product IVP solution
"""

# If saved as json, use below
with gzip.open('results/sample_network', 'r') as fin:        # 4. gzip
    json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
json_loaded = json.loads(json_str)
loaded_network = TemporalNetworkDecoder().decode(json_str=json_str)

# Otherwise, just re-create the sample network
# a_temporal_network = synthetic_demo1(5, .0017)
# loaded_network = a_temporal_network

"""
Experiment for comparing error measures
"""

current_net_3 = loaded_network
current_net_1 = loaded_network
current_net_norm2 = loaded_network

# Handling the temporal solution for integral of error to temporal solution
temp_t, temp_inf, temp_net = run_temporal(loaded_network, loaded_network.snapshots[0].duration,
                                          loaded_network.snapshots[0].beta, loaded_network.length)

integrals_order3 = np.zeros(49)
integrals_order1 = np.zeros(49)
integrals_norm2 = np.zeros(49)

sns.set_palette('Set1')
for x in range(49):
    print(x)
    new_net_3, _ = Compressor._compress_round(current_net_3, 'optimal', 'combined', 3, None)
    new_net_1, _ = Compressor._compress_round(current_net_1, 'optimal', 'combined', 1, None)
    new_net_norm2, _ = Compressor._compress_round(current_net_norm2, 'optimal', 'combined', 1, 2)
    snapshots_3 = new_net_3.snapshots
    snapshots_1 = new_net_1.snapshots
    snapshots_norm2 = new_net_norm2.snapshots
    snapshot_xi_order3 = np.zeros(len(snapshots_3) - 1)
    snapshot_xi_order1 = np.zeros(len(snapshots_1) - 1)
    snapshot_xi_norm2 = np.zeros(len(snapshots_norm2) - 1)
    for i in range(len(snapshots_3) - 1):
        snap_A = snapshots_3[i]
        snap_B = snapshots_3[i + 1]
        xi_AB_3 = Compressor.epsilon(snap_A, snap_B, error_type='combined', order=3, norm=None)
        snapshot_xi_order3[i] = xi_AB_3
    for i in range(len(snapshots_1) - 1):
        snap_A = snapshots_1[i]
        snap_B = snapshots_1[i + 1]
        xi_AB_1 = Compressor.epsilon(snap_A, snap_B, error_type='combined', order=1, norm=None)
        snapshot_xi_order1[i] = xi_AB_1
    for i in range(len(snapshots_norm2) - 1):
        snap_A = snapshots_norm2[i]
        snap_B = snapshots_norm2[i + 1]
        xi_AB_norm2 = Compressor.epsilon(snap_A, snap_B, error_type='combined', order=1, norm=2)
        snapshot_xi_norm2[i] = xi_AB_norm2
    fig, ax = plt.subplots(2, 1)
    max_3 = np.argsort(snapshot_xi_order3)[-1]
    max_1 = np.argsort(snapshot_xi_order1)[-1]
    max_norm2 = np.argsort(snapshot_xi_norm2)[-1]
    # ax[0].scatter(np.argsort(snapshot_xi_order3), np.argsort(snapshot_xi_order1), alpha=0.5, label='Order 3 v Linear')
    ax[0].scatter(np.argsort(snapshot_xi_order3), np.argsort(snapshot_xi_norm2), alpha=0.5, label='O(3) v O(1) w/ norm')
    # ax[0].scatter(max_3, max_1, color='black', s=100, fc=None, alpha=0.9, label=f'max ranks for O(1): {max_1}, O(3):{max_3}')
    ax[0].scatter(max_3, max_norm2, color='black', s=100, fc=None, alpha=0.9, label=f'max ranks for O(3): {max_3}, O(1)+norm:{max_norm2}')
    # ax[0].scatter(max_norm2, max_1, color='yellow', s=100, fc=None, alpha=1.0, label=f'for O(1)w/Norm: {max_norm2}')
    ax[0].set_xlabel('Rank of error $\\xi(s, s+1)$ using O(3)')
    # ax[0].set_ylabel('Rank using O(1)') # and O(1) & Norm')
    ax[0].set_ylabel('Rank using O(1) w/ norm') # and O(1) & Norm')
    ax[1].scatter(np.arange(len(snapshot_xi_order3)), snapshot_xi_order3, alpha=0.5, label='order 3')
    ax[1].plot(np.arange(len(snapshot_xi_order3)), snapshot_xi_order3, alpha=0.5)
    # ax[1].scatter(np.arange(len(snapshot_xi_order1)), snapshot_xi_order1, alpha=0.5, label='order 1')
    # ax[1].plot(np.arange(len(snapshot_xi_order1)), snapshot_xi_order1, alpha=0.5)
    ax[1].scatter(np.arange(len(snapshot_xi_norm2)), snapshot_xi_norm2, alpha=0.5, label='O(1) w/ norm 2')
    ax[1].plot(np.arange(len(snapshot_xi_norm2)), snapshot_xi_norm2, alpha=0.5)
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel('Snapshot number $s$ (time increments)')
    ax[1].set_ylabel('$\\xi(s, s+1)$')
    plt.show()

    ## Solving for the 3 different compressions and comparing the integrated error
    # if x % 5 == 0:
    t_3, I_3, _ = solve_on_given_network(new_net_3, new_net_3.snapshots[0].beta)
    t_1, I_1, _ = solve_on_given_network(new_net_1, new_net_1.snapshots[0].beta)
    t_n2, I_n2, _ = solve_on_given_network(new_net_norm2, new_net_norm2.snapshots[0].beta)
    integral_3 = integrate_error_ts(temp_t, temp_inf, t_3, I_3)  # integral of normalized distances from temporal time series
    integral_1 = integrate_error_ts(temp_t, temp_inf, t_1, I_1)  # integral of normalized distances from temporal time series
    integral_n2 = integrate_error_ts(temp_t, temp_inf, t_n2, I_n2)  # integral of normalized distances from temporal time series
    integrals_order3[x] = integral_3
    integrals_order1[x] = integral_1
    integrals_norm2[x] = integral_n2
    # plt.plot(temp_t, temp_inf)
    # plt.plot(t_3, I_3, alpha=0.4, label="I_3")
    # plt.plot(t_1, I_1, alpha=0.4, label="I_1")
    # plt.plot(t_n2, I_n2, alpha=0.4, label="I_n2")
    # plt.legend()
    # plt.show()
    current_net_3 = new_net_3
    current_net_1 = new_net_1
    current_net_norm2 = new_net_norm2

# results = np.array([integrals_order3, integrals_order1, integrals_norm2])
# np.savetxt('results/integrals_compare.txt', results, delimiter=',')
results = np.loadtxt('results/integrals_compare.txt', delimiter=',')
integrals_order3 = results[0]
integrals_order1 = results[1]
integrals_norm2 = results[2]
plt.scatter(np.arange(49), integrals_order3, label='order 3', alpha=0.5)
plt.scatter(np.arange(49), integrals_order1, label='order 1', alpha=0.5)
plt.scatter(np.arange(49), integrals_norm2, label='order 1 norm 2', alpha=0.5)
plt.legend()
plt.semilogy()
plt.xlabel('Number of compressions')
plt.ylabel('Integrated error from temp solution')
plt.show()



