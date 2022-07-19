import matplotlib.pyplot as plt
import numpy as np

from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json
import gzip

"""
Panel (a) and (b) for Fig. 3 experiment.
Solving single round of 50 snapshots -> 6 snapshots
"""
# a_temporal_network = synthetic_demo1(5, .0017)
# encoded_network = TemporalNetworkEncoder().encode(a_temporal_network)
# fp = open('encoded_network.json', 'r')
# loaded_network = json.load(fp)
# loaded_bytes = encoded_network.encode('utf-8')
# fp.close()
# with gzip.open('sample_network', 'w') as fout:       # 4. fewer bytes (i.e. gzip)
#     fout.write(loaded_bytes)
# text_file = open("encoded_network.txt", "w")
# n = text_file.write(encoded_network)
# text_file.close()
# fp = open('encoded_network.json', 'r')
with gzip.open('results/sample_network', 'r') as fin:        # 4. gzip
    json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
json_loaded = json.loads(json_str)
loaded_network = TemporalNetworkDecoder().decode(json_str=json_str)
# loaded_network = TemporalNetworkDecoder().decode(fname='encoded_network.json')
# fp.close()
# plot_edge_hist(a_temporal_network.snapshots, 5)
# plot_edge_hist(loaded_network.snapshots, 5)
# plt.show()
# results_one_round = one_round(a_temporal_network, 5, .0017, a_temporal_network.length, iters=44)
# results_one_round = one_round(loaded_network, 5, .0017, loaded_network.length, iters=44)
# results_one_round = one_round(loaded_network, 5, .0017, loaded_network.length, iters=44, order=1, norm=2)
# plot_one_round(results_one_round)
# f = open('synthetic_data.json', "w")
# json.dump(results_one_round, f)
# f.close()
# Loading pre-saved results:
results_one_round = json.load(open('results/synthetic_data.json', "r"))
# results_one_round = one_round(loaded_network, 5, .0017, loaded_network.length, iters=40, order=1, norm=None)
# plot_one_round(results_one_round)
# plt.show()
######

"""
Panel (c) for Fig. 3 experiment.
Finds normalized compression error against the fully temporal solution for even splits and algorithmic
as a function of the final number of snapshots in the compressed network.
Initial 50 snapshots, compressions from 30 to 50 steps.
"""
a_temporal_network = loaded_network
# results = error_as_fn_of(a_temporal_network, .0017, np.arange(30, 50, 1))
# result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
# np.savetxt('./results/synthetic_error_integrals.txt', result_array, delimiter=',')
result_array = np.loadtxt('./results/synthetic_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
plot_error_fn_compressions(results)
plt.show()
plot_manuscript_demo({"one_round_results": results_one_round, "error_difference_results": results})
plt.show()

"""
Robustness experiment for range of beta.
Finds the difference between even compression error against temporal solution and algorithmic compression error.
For a range of beta.
"""
iter_range = np.arange(30,50,1)
betas = np.array([.0008, .0009, .001, .0012, .0015, .0017, .002, .0022, .0024, .0026, .0029])
tce_totals = np.zeros(len(iter_range))
# results = total_error_as_fn_of(betas, iter_range, a_temporal_network)
# np.savetxt('synthetic_difference_2.txt', results["difference_matrix"], delimiter=',')
# np.savetxt('synthetic_tce_2.txt', results["tce"], delimiter=',')
results = {}
results["difference_matrix"] = np.loadtxt('./results/synthetic_difference_2.txt', delimiter=',')
results["tce"] = np.loadtxt('./results/synthetic_tce_2.txt', delimiter=',')

plt.scatter(iter_range[1:], results["tce"][4][1:], marker='^', color='k', s=9)
plt.box(False)
plt.xlabel('Number of compressions')
plt.semilogy()
plt.tight_layout()
plt.show()

plot_comparison_and_error(results, betas, iter_range,
                          total_time=np.sum([snapshot.duration for snapshot in a_temporal_network.snapshots]),
                          orig_net_length=50)
plt.show()
plot_heatmap(results["difference_matrix"], betas, iter_range, "Difference algo-even")
plt.show()
plot_heatmap(results["tce"], betas, iter_range, "Total chosen error")
plt.show()
plot_tce(results["tce"], betas, iter_range)
plt.show()
