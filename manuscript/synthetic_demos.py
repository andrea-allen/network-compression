from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *


a_temporal_network = synthetic_demo1(5, .001)
# results = one_round(a_temporal_network, 5, .001, a_temporal_network.length, iters=44)
# plot_one_round(results)
# a_temporal_network.set_all_betas(.004)
# results = one_round(a_temporal_network, 5, .004, a_temporal_network.length, iters=44)
# plot_one_round(results)
# plt.show()

# results = error_as_fn_of(a_temporal_network, .001, np.arange(40, 50, 1))
# plot_error_fn_compressions(results)
# plt.show()

iter_range = np.arange(30,50,2)
betas = np.array([.0008, .0009, .001, .0012, .0015, .0017, .002])
# results = total_error_as_fn_of(betas, iter_range, a_temporal_network)
# np.savetxt('synthetic_difference.txt', results["difference_matrix"], delimiter=',')
# np.savetxt('synthetic_tce.txt', results["tce"], delimiter=',')
results = {}
results["difference_matrix"] = np.loadtxt('synthetic_difference.txt', delimiter=',')
results["tce"] = np.loadtxt('synthetic_tce.txt', delimiter=',')
plot_comparison_and_error(results, betas, iter_range)
plt.show()
# plot_heatmap(results["difference_matrix"], betas, iter_range, "Difference algo-even")
# plt.show()
plot_heatmap(results["tce"], betas, iter_range, "Total chosen error")
plt.show()
plot_tce(results["tce"], betas, iter_range)
plt.show()
