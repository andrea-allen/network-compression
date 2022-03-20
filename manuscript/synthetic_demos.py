from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *


a_temporal_network = synthetic_demo1(5, .001)
results = one_round(a_temporal_network, 5, .001, a_temporal_network.length, iters=44)
plot_one_round(results)
results = one_round(a_temporal_network, 5, .001, a_temporal_network.length, iters=30)
plot_one_round(results)
plt.show()