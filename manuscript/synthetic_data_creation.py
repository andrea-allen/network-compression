### Creating synthetic data for part of my supplement
# Compressing using the new changes to the algorithm 7/19

import matplotlib.pyplot as plt
import numpy as np

from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import json
import gzip

a_temporal_network = synthetic_demo1(5, .0017)
results_one_round = one_round(a_temporal_network, 5, .0017,
                              a_temporal_network.length, iters=40,
                              order=1, norm=2,
                              save_metadata=True)
plot_one_round(results_one_round)
plt.show()