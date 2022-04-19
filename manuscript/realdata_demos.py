import manuscript.network_generators as netgen
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json

#Conference data:
total_time = 1016440 - 28820
num_layers = 200
beta = .000025
# conference_snapshots = netgen.data_network_snapshots('../raw_data/tij_InVS.dat', int(total_time/num_layers), beta, int(num_layers/2), 28820)

# results = one_round(TemporalNetwork(conference_snapshots), int(total_time/num_layers), beta, len(conference_snapshots), iters=180)
# plot_one_round(results)
# for snap in conference_snapshots:
#     snap.beta = .00003
# results = one_round(TemporalNetwork(conference_snapshots), int(total_time/num_layers), .00003, len(conference_snapshots), iters=180)
# plot_one_round(results)
# plt.show()

# results = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_layers), beta, len(hospital_snapshots), iters=180)
# plot_one_round(results)
total_time = 347640 - 140
# netgen.dataset_statistics('../raw_data/detailed_list_of_contacts_Hospital.dat_')
# plt.show()
# Durations with 200 layers: 1737
hospital_snapshots = netgen.data_network_snapshots('../raw_data/detailed_list_of_contacts_Hospital.dat_', int(total_time/num_layers), .000015, int(num_layers/2), 140)

# plot_edge_hist(hospital_snapshots)
# plt.show()
# for snap in hospital_snapshots:
#     snap.beta = .00002
# results = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_layers), .000025, len(hospital_snapshots), iters=192)
# plot_one_round(results)
# results = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_layers), .000015, len(hospital_snapshots), iters=188)
# plot_one_round(results)
# results_one_round = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_layers), .000015, len(hospital_snapshots), iters=190)
# f = open('hospital_data.json', "w")
# json.dump(results_one_round, f)
# f.close()
results_one_round = json.load(open('hospital_data.json', "r"))
plot_one_round(results_one_round)
# for snap in hospital_snapshots:
#     snap.beta = .000035
# results = one_round(TemporalNetwork(hospital_snapshots), int(total_time/num_layers), .000035, len(hospital_snapshots), iters=192)
# plot_one_round(results)
plt.show()
# results = error_as_fn_of(TemporalNetwork(hospital_snapshots), .000015, np.arange(100, 200, 1))
# result_array = np.array([results['iter_range'], results['even_error_norm'], results['opt_error_norm'], results['tce_all']])
# np.savetxt('hospital_error_integrals.txt', result_array, delimiter=',')
# plot_error_fn_compressions(results)
# plt.show()

## for paper:
result_array = np.loadtxt('hospital_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0][70:], 'even_error_norm': result_array[1][70:], 'opt_error_norm': result_array[2][70:], 'tce_all': result_array[3][70:]}
plot_error_fn_compressions(results)
plt.show()
plot_manuscript_demo({"one_round_results": results_one_round, "error_difference_results": results})
# plt.savefig('hospital_validation_04-08-22.png')
# plt.savefig('hospital_validation_04-08-22.svg', fmt='svg')
plt.show()
####

iter_range = np.arange(150,200,1)
betas = np.array([.000005, .00001, .000015, .00002, .000025, .00003, .000035])
# results = total_error_as_fn_of(betas, iter_range, TemporalNetwork(hospital_snapshots))
# np.savetxt('hospital_difference.txt', results["difference_matrix"], delimiter=',')
# np.savetxt('hospital_tce.txt', results["tce"], delimiter=',')
results = {}
results["difference_matrix"] = np.loadtxt('hospital_difference.txt', delimiter=',')
results["tce"] = np.loadtxt('hospital_tce.txt', delimiter=',')
plt.scatter(iter_range[1:], results["tce"][4][1:], marker='^', color='k', s=9)
plt.box(False)
plt.xlabel('Number of compressions')
plt.semilogy()
plt.tight_layout()
plt.show()
plot_comparison_and_error(results, betas, iter_range, total_time, orig_net_length=200)
plt.show()
