import numpy as np
import scipy
from scipy import integrate


class TemporalSIModel:
    def __init__(self, params, y_init, end_time, networks, approximate=False):
        self.params = params
        self.y_init = y_init
        self.y_current = self.y_init
        self.result = []
        self.end_time = end_time
        self.start_time = 0
        self.approximate = approximate
        self.networks = networks ## dictionary of switch times to adjacency matrices
        self.switch_times = sorted(list(self.networks.keys()))
        self.current_switch_time_index = 0
        self.current_switch_time = self.switch_times[0]
        self.N = len(y_init)

    def odes_si(self, y, t):
        beta = self.params['beta']
        derivatives = np.zeros(len(y))
        A = self.networks[self.current_switch_time]
        for i in range(self.N):
            derivatives[i] = (1 - y[i]) * beta * np.sum(A[i] @ y)
            if self.approximate:
                derivatives[i] = (1) * beta * np.sum(A[i] @ y)
        return derivatives

    def solve_model(self, total_time_steps=300, return_p_vecs = False, custom_t_inc = None):
        time_series_result = []
        node_probabilities = [[] for n in range(self.N)]
        steps_per_interval = max(int(np.round(total_time_steps/len(self.switch_times), 0)),20)
        p_states = [self.y_current]
        for switchtime in self.switch_times:
            self.current_switch_time = switchtime
            if custom_t_inc is None:
                solve_for_timesteps = np.linspace(self.start_time, switchtime, steps_per_interval)
            else:
                solve_for_timesteps = np.arange(self.start_time, switchtime, custom_t_inc)
            switchtime_solution = scipy.integrate.odeint(self.odes_si,
                                                         y0=self.y_current,
                                                         t=solve_for_timesteps)
            time_series_result.extend(solve_for_timesteps)
            for i in range(self.N):
                node_probabilities[i].extend(list(switchtime_solution[:, i]))
            new_initial_p_vec = [switchtime_solution[:,i][-1] for i in range(self.N)]
            self.start_time = switchtime
            self.y_current = new_initial_p_vec
            p_states.append(self.y_current)
        if return_p_vecs:
            return time_series_result, np.array(node_probabilities), p_states
        return time_series_result, np.array(node_probabilities)


def digitize_solution(time_vector, infected, number_layers, t_interval):
    digitized = np.digitize(np.array(time_vector), np.linspace(0,number_layers*t_interval,50))
    bin_means = [np.array(time_vector)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    bins_infected = [np.array(infected)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    # interpolate the nans with backfill:
    for i, d in enumerate(bin_means):
        if np.isnan(d):
            bin_means[i] = bin_means[i-1]
    for i, d in enumerate(bins_infected):
        if np.isnan(d):
            bins_infected[i] = bins_infected[i-1]
    return bin_means, bins_infected