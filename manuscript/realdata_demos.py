import manuscript.network_generators as netgen
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *

#Conference data:
total_time = 1016440 - 28820
num_layers = 200
beta = .00002
conference_snapshots = netgen.data_network_snapshots('../raw_data/tij_InVS.dat', int(total_time/num_layers), beta, int(num_layers/2), 28820)

results = one_round(TemporalNetwork(conference_snapshots), int(total_time/num_layers), beta, len(conference_snapshots), iters=180)
plot_one_round(results)
for snap in conference_snapshots:
    snap.beta = .00003
results = one_round(TemporalNetwork(conference_snapshots), int(total_time/num_layers), .00003, len(conference_snapshots), iters=180)
plot_one_round(results)
plt.show()
