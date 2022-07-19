import pandas as pd
import manuscript.network_generators as netgen
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json
import constants

"""
On empirical dataset (pre-saved sample network used in manuscript, hospital network)
Testing whether the linear approximation works as well as the O(3) approximation
Testing whether using the operator norm works as well as the dot product IVP solution
"""

"""
Dataset network statistics for hospital data
"""
# netgen.dataset_statistics('../raw_data/detailed_list_of_contacts_Hospital.dat_')
# plt.show()

####################################

# Durations with 200 snapshots: 1737
num_snapshots = 200
hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_',
                                                   int(constants.HOSP_TOTAL_TIME / num_snapshots), .000015,
                                                   int(num_snapshots / 2),
                                                   constants.HOSP_START_TIME)

####
loaded_network = TemporalNetwork(hospital_snapshots)
# Handling the temporal solution for integral of error to temporal solution
temp_t, temp_inf, temp_net = run_temporal(loaded_network, loaded_network.snapshots[0].duration,
                                          loaded_network.snapshots[0].beta, loaded_network.length)

integrals_order3 = np.zeros(196)
integrals_order1 = np.zeros(196)
integrals_norm2 = np.zeros(196)

sns.set_palette('Set1')
current_net_3 = loaded_network
current_net_1 = loaded_network
current_net_norm2 = loaded_network

## Uncomment to perform full experiment again
# for x in range(196):
#     print(x)
#     new_net_3, _ = Compressor._compress_round(current_net_3, 'optimal', 'combined', 3, None)
#     new_net_1, _ = Compressor._compress_round(current_net_1, 'optimal', 'combined', 1, None)
#     new_net_norm2, _ = Compressor._compress_round(current_net_norm2, 'optimal', 'combined', 1, 2)
#     # if x % 10 == 0 or x > 190:
#     #     snapshots_3 = new_net_3.snapshots
#     #     snapshots_1 = new_net_1.snapshots
#     #     snapshots_norm2 = new_net_norm2.snapshots
#     #     snapshot_xi_order3 = np.zeros(len(snapshots_3) - 1)
#     #     snapshot_xi_order1 = np.zeros(len(snapshots_1) - 1)
#     #     snapshot_xi_norm2 = np.zeros(len(snapshots_norm2) - 1)
#     #     for i in range(len(snapshots_3) - 1):
#     #         snap_A = snapshots_3[i]
#     #         snap_B = snapshots_3[i + 1]
#     #         xi_AB_3 = Compressor.epsilon(snap_A, snap_B, error_type='combined', order=3, norm=None)
#     #         snapshot_xi_order3[i] = xi_AB_3
#     #     for i in range(len(snapshots_1) - 1):
#     #         snap_A = snapshots_1[i]
#     #         snap_B = snapshots_1[i + 1]
#     #         xi_AB_1 = Compressor.epsilon(snap_A, snap_B, error_type='combined', order=1, norm=None)
#     #         snapshot_xi_order1[i] = xi_AB_1
#     #     for i in range(len(snapshots_norm2) - 1):
#     #         snap_A = snapshots_norm2[i]
#     #         snap_B = snapshots_norm2[i + 1]
#     #         xi_AB_norm2 = Compressor.epsilon(snap_A, snap_B, error_type='combined', order=1, norm=2)
#     #         snapshot_xi_norm2[i] = xi_AB_norm2
#     #     fig, ax = plt.subplots(2, 1)
#     #     max_3 = np.argsort(snapshot_xi_order3)[-1]
#     #     max_1 = np.argsort(snapshot_xi_order1)[-1]
#     #     max_norm2 = np.argsort(snapshot_xi_norm2)[-1]
#     #     ax[0].scatter(np.argsort(snapshot_xi_order3), np.argsort(snapshot_xi_order1), label='O(3) v O(1)')
#     #     ax[0].scatter(np.argsort(snapshot_xi_order3), np.argsort(snapshot_xi_norm2), label='O(3) v O(1) w/ norm')
#     #     ax[0].scatter(max_3, max_1, color='red', s=100, fc=None, alpha=0.9, label=f'max rank O(3): {max_3}')
#     #     ax[0].scatter(max_norm2, max_1, color='yellow', s=100, fc=None, alpha=1.0,
#     #                   label=f'max ranks O(1): {max_1}, O(1) w norm: {max_norm2}')
#     #     ax[0].set_xlabel('Rank of error $\\xi(s, s+1)$ using O(3)')
#     #     ax[0].set_ylabel('Rank using O(1) and O(1) & Norm')
#     #     ax[1].scatter(np.arange(len(snapshot_xi_order3)), snapshot_xi_order3, alpha=0.5, label='O(3)')
#     #     ax[1].plot(np.arange(len(snapshot_xi_order3)), snapshot_xi_order3, alpha=0.5)
#     #     ax[1].scatter(np.arange(len(snapshot_xi_order1)), snapshot_xi_order1, alpha=0.5, label='O(1)')
#     #     ax[1].plot(np.arange(len(snapshot_xi_order1)), snapshot_xi_order1, alpha=0.5)
#     #     ax[1].scatter(np.arange(len(snapshot_xi_norm2)), snapshot_xi_norm2, alpha=0.5, label='O(1) w/ norm')
#     #     ax[1].plot(np.arange(len(snapshot_xi_norm2)), snapshot_xi_norm2, alpha=0.5)
#     #     ax[0].legend()
#     #     ax[1].legend()
#     #     ax[1].set_xlabel('Snapshot number $s$ (time increments)')
#     #     ax[1].set_ylabel('$\\xi(s, s+1)$')
#     #     plt.show()
#
#     ## Solving for the 3 different compressions and comparing the integrated error
#     if x % 10 == 0 or x > 185:
#         t_3, I_3, _ = solve_on_given_network(new_net_3, new_net_3.snapshots[0].beta)
#         t_1, I_1, _ = solve_on_given_network(new_net_1, new_net_1.snapshots[0].beta)
#         t_n2, I_n2, _ = solve_on_given_network(new_net_norm2, new_net_norm2.snapshots[0].beta)
#         integral_3 = integrate_error_ts(temp_t, temp_inf, t_3,
#                                         I_3)  # integral of normalized distances from temporal time series
#         integral_1 = integrate_error_ts(temp_t, temp_inf, t_1,
#                                         I_1)  # integral of normalized distances from temporal time series
#         integral_n2 = integrate_error_ts(temp_t, temp_inf, t_n2,
#                                          I_n2)  # integral of normalized distances from temporal time series
#         integrals_order3[x] = integral_3
#         integrals_order1[x] = integral_1
#         integrals_norm2[x] = integral_n2
#         # plt.plot(temp_t, temp_inf)
#         # plt.plot(t_3, I_3, alpha=0.4, label="I_3")
#         # plt.plot(t_1, I_1, alpha=0.4, label="I_1")
#         # plt.plot(t_n2, I_n2, alpha=0.4, label="I_n2")
#         # plt.legend()
#         # plt.show()
#     current_net_3 = new_net_3
#     current_net_1 = new_net_1
#     current_net_norm2 = new_net_norm2
# results = np.array([integrals_order3, integrals_order1, integrals_norm2])
# np.savetxt('results/integrals_compare_data.txt', results, delimiter=',')

results = np.loadtxt('results/integrals_compare_data.txt', delimiter=',')
integrals_order3 = results[0]
integrals_order1 = results[1]
integrals_norm2 = results[2]
plt.scatter(np.arange(len(integrals_order3)), integrals_order3, label='order 3', alpha=0.5)
plt.scatter(np.arange(len(integrals_order3)), integrals_order1, label='order 1', alpha=0.5)
plt.scatter(np.arange(len(integrals_order3)), integrals_norm2, label='order 1 norm 2', alpha=0.5)
plt.legend()
plt.xlabel('Number of compressions')
plt.ylabel('Integrated error from temp solution')
plt.show()