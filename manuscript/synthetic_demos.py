from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *


a_temporal_network = synthetic_demo1(5, .001)
results = one_round(a_temporal_network, 5, .001, a_temporal_network.length, iters=44)
plot_one_round(results)
a_temporal_network.set_all_betas(.004)
results = one_round(a_temporal_network, 5, .004, a_temporal_network.length, iters=44)
plot_one_round(results)
plt.show()

results = error_as_fn_of(a_temporal_network, .001, np.arange(40, 50, 1))
plot_error_fn_compressions(results)
plt.show()