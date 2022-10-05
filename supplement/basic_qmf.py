import networkx as nx
import numpy as np
from scipy.linalg import expm
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import matplotlib
import seaborn as sns
# plt.rcParams
# matplotlib.rcParams.update({'font.size': 12})

sns.set_palette("Set2")

# for values tau in 0, T run a deterministic temporal
def solve_QMF(A, beta, end_time):
    snapshot = Snapshot(0, end_time, beta, A)
    y_init = snapshot.dd_normalized
    temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=end_time,
                                 networks={end_time: A})
    solution_t_temporal, solution_p = temp_model.solve_model()
    temporal_timeseries = np.sum(solution_p, axis=0)
    return solution_t_temporal, temporal_timeseries

N = 100
G3, A3 = configuration_model_graph(N)
G6, A6 = erdos_renyi_graph(N, .99)

tau = 50*.003/2
# t_conf, I_conf = solve_QMF(A3, .03, 50)
# Complete graph can be a first thing to compare against the basic SIR, but wait, you can change it
t_ER, I_ER = solve_QMF(A6, .003, 50)
plt.plot(t_ER, N-I_ER, label="Susceptible")
plt.plot(t_ER, I_ER, label="Infected")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Numbr of nodes")
plt.show()

# t_conf, I_conf = solve_QMF(A3, .03, 50)
# Complete graph can be a first thing to compare against the basic SIR, but wait, you can change it
t_C, I_C = solve_QMF(A3, .003, 500)
plt.plot(t_C, N-I_C, label="Susceptible")
plt.plot(t_C, I_C, label="Infected")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Numbr of nodes")
plt.show()


