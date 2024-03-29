from src.compression import *
from src.solvers import *
import matplotlib.pyplot as plt
import json
import numpy as np


def integrate_error_ts(temporal_ts, temporal_inf, other_ts, other_inf):
    other_inf = np.array(other_inf)
    temporal_inf = np.array(temporal_inf)
    if other_inf[0] == 0 or temporal_inf[0] == 0:
        print('correcting')
        other_inf = np.array(other_inf) + .0000000001
        temporal_inf = np.array(temporal_inf) + .0000000001
    absolute_diff_normed = np.abs(other_inf[1:] - temporal_inf[1:]) / temporal_inf[1:]
    time_delta = np.diff(np.array(temporal_ts))
    integrand = np.array([time_delta[i] * absolute_diff_normed[i] for i in range(len(time_delta))])
    total_error_integrand = np.sum(integrand)
    return total_error_integrand


def run_even(temporal_network, t_interval, beta, number_snapshots, iters):
    N = len(temporal_network.snapshots[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.snapshots[0].dd_normalized
    even_compressed = Compressor.compress(temporal_network, iterations=iters,
                                          how='even')
    model = TemporalSIModel(params={'beta': beta}, y_init=y_init,
                            end_time=number_snapshots * t_interval,
                            networks=even_compressed.get_time_network_map())
    solution_t_even, solution_p = model.solve_model()
    even_solution = np.sum(solution_p, axis=0)
    d = digitize_solution(solution_t_even, even_solution, number_snapshots, t_interval)
    return d[0], d[1], even_compressed


def run_optimal(temporal_network, t_interval, beta, number_snapshots, iters, error_type, order, norm,
                save_metadata=False):
    N = len(temporal_network.snapshots[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.snapshots[0].dd_normalized
    optimal_network, total_chosen_error = Compressor.compress(temporal_network, iterations=iters, how='optimal',
                                                              error_type=error_type, order=order, norm=norm)
    # Saving the compressed temporal snapshots from the algorithm
    if save_metadata:
        metadata = {'snapshots': [{'beta': snap.beta, 'duration': snap.duration, 'start_time': snap.start_time,
                                 'end_time': snap.end_time} for snap in optimal_network.snapshots]}
        f = open('results/compressed_synthetic/metadata.json', "w")
        json.dump(metadata, f)
        f.close()
        # Adjacency matrices each get saved to a file
        for snapshot in optimal_network.snapshots:
            filename = f'adjmat_start_{snapshot.start_time}.txt'
            # multiply each adjacency matrix by the duration to make it an integer
            scaled_matrix = snapshot.A * snapshot.duration
            scaled_matrix = scaled_matrix.astype(int)
            np.savetxt(f'results/compressed_synthetic/{filename}', scaled_matrix, fmt='%.7e')
    model = TemporalSIModel(params={'beta': beta}, y_init=y_init,
                            end_time=number_snapshots * t_interval,
                            networks=optimal_network.get_time_network_map())
    solution_t_compressed, solution_p = model.solve_model()
    compressed_solution = np.sum(solution_p, axis=0)
    d = digitize_solution(solution_t_compressed, compressed_solution, number_snapshots, t_interval)
    return d[0], d[1], optimal_network, total_chosen_error


def solve_on_given_network(network, beta):
    N = len(network.snapshots[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.snapshots[0].dd_normalized
    model = TemporalSIModel(params={'beta': beta}, y_init=y_init,
                            end_time=network.snapshots[-1].end_time,
                            networks=network.get_time_network_map())
    solution_t_compressed, solution_p = model.solve_model()
    compressed_solution = np.sum(solution_p, axis=0)
    number_snapshots = network.length
    t_interval = network.snapshots[-1].end_time / number_snapshots
    d = digitize_solution(solution_t_compressed, compressed_solution, number_snapshots, t_interval)
    return d[0], d[1], network


def run_temporal(temporal_network, t_interval, beta, number_snapshots):
    N = len(temporal_network.snapshots[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.snapshots[0].dd_normalized
    model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=temporal_network.snapshots[-1].end_time,
                            networks=temporal_network.get_time_network_map())
    # plt.plot([np.sum(model.networks[key]) for key, val in model.networks.items()]) # 2x contacts per snapshot, plotted
    # plt.show()
    solution_t_temporal, solution_p = model.solve_model()
    temporal_solution = np.sum(solution_p, axis=0)
    total_time = temporal_network.snapshots[-1].end_time - temporal_network.snapshots[0].start_time
    t_interval = total_time/number_snapshots
    d = digitize_solution(solution_t_temporal, temporal_solution, number_snapshots, t_interval)
    return d[0], d[1], temporal_network


def one_round(temporal_network, t_interval, beta, number_snapshots, iters, order, norm, save_metadata=False):
    temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_snapshots)
    even_t, even_inf, even_net = run_even(temporal_network, t_interval, beta, number_snapshots, iters)
    opt_t_c, opt_inf_c, opt_net_c, total_chosen_error = run_optimal(temporal_network, t_interval, beta,
                                                                    number_snapshots, iters, 'combined',
                                                                    order, norm,
                                                                    save_metadata)
    total_optimal_c_error_nm = round(integrate_error_ts(temp_t, temp_inf, opt_t_c, opt_inf_c), 3)
    total_even_error_nm = round(integrate_error_ts(temp_t, temp_inf, even_t, even_inf), 3)

    results = {'temp_t': temp_t, 'temp_inf': temp_inf, 'algo_t': opt_t_c, 'algo_inf': opt_inf_c, 'even_t': even_t,
               'even_inf': even_inf, 'algo_boundary_times': list(opt_net_c.get_time_network_map().keys()),
               'temp_boundary_times': list(temporal_network.get_time_network_map().keys()),
               'even_boundary_times': list(even_net.get_time_network_map().keys()),
               'total_algo_error': total_optimal_c_error_nm, 'total_even_error': total_even_error_nm,
               'total_chosen_error': total_chosen_error}

    return results

def compare_mdl(temporal_network_algo, temporal_network_mdl, temporal_net_original,
                t_interval, beta, iters):
    temp_t, temp_inf, temp_net = run_temporal(temporal_net_original, t_interval, beta, temporal_net_original.length)
    opt_t, opt_inf, opt_net = run_temporal(temporal_network_algo, t_interval, beta, temporal_network_algo.length)
    mdl_t, mdl_inf, mdl_net = run_temporal(temporal_network_mdl, t_interval, beta, temporal_network_mdl.length)
    even_t, even_inf, even_net = run_even(temporal_net_original, t_interval, beta, temporal_net_original.length, iters)
    total_optimal_c_error_nm = round(integrate_error_ts(temp_t, temp_inf, opt_t, opt_inf), 3)
    total_mdl_error_nm = round(integrate_error_ts(temp_t, temp_inf, mdl_t, mdl_inf), 3)
    total_even_error_nm = round(integrate_error_ts(temp_t, temp_inf, even_t, even_inf), 3)

    results = {'temp_t': temp_t, 'temp_inf': temp_inf, 'algo_t': opt_t, 'algo_inf': opt_inf,
               'even_t': even_t,'even_inf': even_inf, 'mdl_t': mdl_t, 'mdl_inf': mdl_inf,
               'algo_boundary_times': list(opt_net.get_time_network_map().keys()),
               'temp_boundary_times': list(temporal_net_original.get_time_network_map().keys()),
               'even_boundary_times': list(even_net.get_time_network_map().keys()),
               'mdl_boundary_times': list(mdl_net.get_time_network_map().keys()),
               'total_algo_error': total_optimal_c_error_nm,
               'total_even_error': total_even_error_nm,
               'total_mdl_error': total_mdl_error_nm,
               'total_chosen_error': 0}

    return results


def error_as_fn_of(temp_net, beta, iter_range):
    t_interval = temp_net.snapshots[0].duration  # random snapshot, should be same t_interval for all starting out
    if iter_range is None:
        gap = 5
        iter_range = np.arange(0, temp_net.length, gap)
    even_errors_norm = np.zeros(len(iter_range))
    optimal_errors_norm = np.zeros(len(iter_range))
    tce_all = np.zeros(len(iter_range))
    temp_t, temp_inf, temp_net = run_temporal(temp_net, t_interval, beta, temp_net.length)

    current_optimal_temp_net = temp_net
    current_iters_for_optim = 0
    for i, r in enumerate(iter_range):
        c = int(r)
        even_t, even_inf, even_net = run_even(temp_net, t_interval, beta, temp_net.length, c)
        opt_t, opt_inf, opt_net, tce = run_optimal(current_optimal_temp_net, t_interval, beta,
                                                   temp_net.length, c - current_iters_for_optim,
                                                   'combined', order=1, norm=2)
        current_iters_for_optim = c
        current_optimal_temp_net = opt_net
        total_optimal_error_nm = integrate_error_ts(temp_t, temp_inf, opt_t, opt_inf) # integral of normalized distances from temporal time series
        print(f"***, {i, r}")
        total_even_error_nm = integrate_error_ts(temp_t, temp_inf, even_t, even_inf)

        optimal_errors_norm[i] = total_optimal_error_nm
        even_errors_norm[i] = total_even_error_nm
        tce_all[i] = tce

    results = {"opt_error_norm": optimal_errors_norm, "even_error_norm": even_errors_norm,
               "tce_all": tce_all, "iter_range": iter_range
               }
    return results


def total_epsilon_over(temp_net, beta, iter_range):
    t_interval = temp_net.snapshots[0].duration  # random snapshot, should be same t_interval for all starting out
    if iter_range is None:
        gap = 5
        iter_range = np.arange(0, temp_net.length, gap)
    tce_all = np.zeros(len(iter_range))
    current_optimal_temp_net = temp_net
    current_iters_for_optim = 0
    for i, r in enumerate(iter_range):
        c = int(r)
        opt_t, opt_inf, opt_net, tce = run_optimal(current_optimal_temp_net, t_interval, beta,
                                                   temp_net.length, c - current_iters_for_optim, 'combined')
        current_iters_for_optim = c
        current_optimal_temp_net = opt_net
        print(f"***, {i, r}")

        tce_all[i] = tce

    results = {
        "tce_all": tce_all, "iter_range": iter_range
    }
    return results


def total_error_as_fn_of(betas, iter_range, a_temporal_network):
    difference_matrix = np.zeros((len(betas), len(iter_range)))
    tce_matrix = np.zeros((len(betas), len(iter_range)))
    for b in range(len(betas)):
        a_temporal_network.set_all_betas(betas[b])
        error_results = error_as_fn_of(a_temporal_network, betas[b], iter_range)
        difference_matrix[b] = error_results["opt_error_norm"] - error_results["even_error_norm"]
        tce_matrix[b] = error_results["tce_all"]
    results = {"difference_matrix": difference_matrix, "tce": tce_matrix}
    return results

def plot_pairwise_error(snapshots):
    # Save the pairwise errors and make a cool plot
    compressor = Compressor()
    start_times = []
    epsilons = []
    for s in range(len(snapshots)-1):
        s1 = snapshots[s]
        s2 = snapshots[s+1]
        e = compressor.epsilon(s1, s2, error_type="combined", order=1, norm=2)
        start_times.append(s1.start_time)
        epsilons.append(e)
    return start_times, epsilons
