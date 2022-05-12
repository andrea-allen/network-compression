import numpy as np

from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json


### POSTER
a_temporal_network = synthetic_demo1(5, .0017)
# plot_edge_hist(a_temporal_network.snapshots, 5)
# plt.show()
# results_one_round = one_round(a_temporal_network, 5, .0017, a_temporal_network.length, iters=44)
# f = open('synthetic_data.json', "w")
# json.dump(results_one_round, f)
# f.close()
results_one_round = json.load(open('results/synthetic_data.json', "r"))
plot_one_round(results_one_round)
# a_temporal_network.set_all_betas(.004)
# results = one_round(a_temporal_network, 5, .004, a_temporal_network.length, iters=44)
# plot_one_round(results)
plt.show()
######

# results = error_as_fn_of(a_temporal_network, .0017, np.arange(30, 50, 1))
# result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
# np.savetxt('synthetic_error_integrals.txt', result_array, delimiter=',')
result_array = np.loadtxt('synthetic_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
plot_error_fn_compressions(results)
plt.show()
plot_manuscript_demo({"one_round_results": results_one_round, "error_difference_results": results})
# plt.savefig('synthetic_validation_04-08-22.png')
# plt.savefig('synthetic_validation_04-08-22.svg', fmt='svg')
plt.show()

## TODO redo one beta of results and save epsilon and see if that works instead
iter_range = np.arange(30,50,1)
betas = np.array([.0008, .0009, .001, .0012, .0015, .0017, .002, .0022, .0024, .0026, .0029])
tce_totals = np.zeros(len(iter_range))
# results = total_error_as_fn_of(betas, iter_range, a_temporal_network)
# np.savetxt('synthetic_difference_2.txt', results["difference_matrix"], delimiter=',')
# np.savetxt('synthetic_tce_2.txt', results["tce"], delimiter=',')
results = {}
results["difference_matrix"] = np.loadtxt('synthetic_difference_2.txt', delimiter=',')
results["tce"] = np.loadtxt('synthetic_tce_2.txt', delimiter=',')

# plt.scatter(iter_range[1:], results["tce"][4][1:], marker='^', color='k', s=9)
# plt.box(False)
# plt.xlabel('Number of compressions')
# plt.semilogy()
# plt.tight_layout()
# plt.show()
plot_comparison_and_error(results, betas, iter_range,
                          total_time=np.sum([snapshot.duration for snapshot in a_temporal_network.snapshots]),
                          orig_net_length=50)
plt.show()
# plot_heatmap(results["difference_matrix"], betas, iter_range, "Difference algo-even")
# plt.show()
plot_heatmap(results["tce"], betas, iter_range, "Total chosen error")
plt.show()
plot_tce(results["tce"], betas, iter_range)
plt.show()
