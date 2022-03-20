import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.linalg import expm
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *

def validation_on(A, B, beta, taus):
    error_approx_terminal = np.zeros(len(taus))
    error_approx_halftime = np.zeros(len(taus))
    error_approx_combo = np.zeros(len(taus))
    matexp_temps = np.zeros(len(taus))
    matexp_temps_h = np.zeros(len(taus))
    matexp_aggs = np.zeros(len(taus))
    matexp_aggs_h = np.zeros(len(taus))
    det_temps = np.zeros(len(taus))
    det_temps_halftime = np.zeros(len(taus))
    det_aggs = np.zeros(len(taus))
    det_aggs_halftime = np.zeros(len(taus))

    type_colors = {'temp': 'grey', 'even': "#FFC626", 'algo': "#00A4D4"}
    tau_color = sns.color_palette('Greys', len(taus))

    fig, ax = plt.subplots(2, 3)
    for t, tau in enumerate(taus):
        print(t, tau)
        print(tau/beta)
        # eps = error(A, B)
        A_lay = Snapshot(0, tau/beta, beta, A)
        B_lay = Snapshot(tau/beta, 2*tau/beta, beta, B)
        epsilon_terminal = Compressor.epsilon(A_lay, B_lay, error_type='terminal')
        epsilon_halftime = Compressor.epsilon(A_lay, B_lay, error_type='halftime')
        epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined')
        error_approx_terminal[t] = epsilon_terminal
        error_approx_halftime[t] = epsilon_halftime
        error_approx_combo[t] = epsilon_combo
    # for values tau in 0, T run a deterministic temporal
        y_init = A_lay.dd_normalized
        temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2*tau/beta,
                                networks={tau/beta: A, 2*tau/beta: B})
        solution_t_temporal, solution_p = temp_model.solve_model()
        temporal_timeseries = np.sum(solution_p, axis=0)
        if t % 10 == 0:
            ax[0, 0].plot(solution_t_temporal, temporal_timeseries, color=tau_color[t], lw=1)
        final_temp = temporal_timeseries[-1]
        det_temps[t] = final_temp
        det_temps_halftime[t] = temporal_timeseries[int(len(temporal_timeseries)/2)]
    # plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
        model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2*tau/beta,
                                networks={2*tau/beta: (A+B)/2})
        solution_t_agg, solution_p = model.solve_model()
        aggregate_timeseries = np.sum(solution_p, axis=0)
        if t % 10 == 0:
            ax[0,0].plot(solution_t_agg, aggregate_timeseries, color=tau_color[t], ls='--', lw=1)
            ax[0,0].vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
                           ymax=temporal_timeseries[-1], color='lime', lw=1)
            ax[0, 0].vlines(tau/beta,
                            ymin=aggregate_timeseries[int(len(aggregate_timeseries)/2)],
                            ymax=temporal_timeseries[int(len(temporal_timeseries)/2)],
                            color='yellow', lw=1)
            ax[0,0].fill_between(solution_t_temporal, aggregate_timeseries, temporal_timeseries, color=tau_color[t], alpha=0.3)
        # plt.show()
        final_agg = aggregate_timeseries[-1]
        det_aggs[t] = final_agg
        det_aggs_halftime[t] = aggregate_timeseries[int(len(aggregate_timeseries)/2)]
        # det_temp = _
        # det_agg = _
    # For values tau in 0, T run the matrix approximation temporal
        P0 = np.full(N, 1 / N)
        P0 = A_lay.dd_normalized
        matexp_temp = np.sum(expm(tau*B).dot(expm(tau*A).dot(P0)))
        matexp_agg = np.sum(expm(2*tau*(B + A)/2).dot(P0))
        matexp_temps[t] = matexp_temp
        matexp_aggs[t] = matexp_agg
        matexp_temps_h[t] = np.sum(expm(tau*A).dot(P0))
        matexp_aggs_h[t] = np.sum(expm(tau*(B + A)/2).dot(P0))

    # sns.kdeplot([k for (n,k) in nx.degree(G3)], label='snapshot 1', color='m', ax=ax[0,2]) # not plotting this right but otherwise great
    ax[0,2].hist([k for (n,k) in nx.degree(G3)],  label='snapshot 1', color='m', bins='auto', density=False)
    ax[0,2].hist([k for (n,k) in nx.degree(G6)],  label='snapshot 2', color='c', bins='auto', density=False)
    # sns.kdeplot([k for (n,k) in nx.degree(G6)], label='snapshot 2', color='c', ax=ax[0,2])
    ax[0,2].legend(frameon=False)
    # diff_matexp = matexp_temp - matexp_agg
    # diff_det = det_temp - det_agg
    # Fig 2a: plot tau vs det_temp, det_agg, matexp_temp, matexp_agg
    ax[0, 1].scatter(taus, det_temps, label='deterministic temp', alpha=0.9, color='c', marker='o', fc='none', s=4)
    ax[0, 1].scatter(taus, det_aggs, label='deterministic agg', alpha=0.6, color='c', marker='x', s=4)
    ax[0, 1].scatter(taus, matexp_temps, label='matexp temp', alpha=0.9, color='grey', marker='o', fc='none', s=4)
    ax[0, 1].scatter(taus, matexp_aggs, label='matexp agg', alpha=0.6, color='grey', marker='x', s=4)
    ax[0, 1].set_xlabel('tau')
    ax[0, 1].set_ylabel('number nodes infected \nafter 2T time')
    ax[0, 1].legend(frameon=False)

    # Fig 2b: plot deterministic error, vs matexp error and error approx
    t_vals = np.array([tau/beta for tau in taus])
    ax[1, 1].scatter(np.abs(det_temps-det_aggs), np.abs(matexp_temps-matexp_aggs), label='difference, matexp',
                     alpha=0.6, color='grey', marker='+')
    ax[1, 1].scatter(np.abs(det_temps-det_aggs), error_approx_terminal/(2*t_vals), label='Epsilon terminal', alpha=0.8,
                     color='y', marker='*')
    ax[1, 0].scatter(np.abs(det_temps_halftime-det_aggs_halftime), error_approx_halftime/(t_vals), label='Epsilon halftime', alpha=0.8,
                     color='blue', marker='*')
    ax[1, 0].scatter(np.abs(det_temps-det_aggs), np.abs(matexp_temps_h-matexp_aggs_h), label='difference, matexp',
                     alpha=0.6, color='grey', marker='+')
    ax[1, 0].set_xlabel('diff nodes infected \nafter 2T - deterministic')
    ax[1, 0].set_ylabel('diff nodes infected \nafter 2T time')

    # ax[1,2].plot(taus, np.abs(det_temps-det_aggs), label='deterministic terminal', ls='-', color='k')
    # ax[1,2].plot(taus, np.abs(det_temps_halftime-det_aggs_halftime), label='deterministic halftime', ls='-', color='y')
    # ax[1,2].plot(taus, np.abs(matexp_temps-matexp_aggs), label='matexp term', ls=':', color='k')
    # ax[1,2].plot(taus, np.abs(matexp_temps_h-matexp_aggs_h), label='matexp half', ls=':', color='y')
    # ax[1,2].plot(taus, error_approx_terminal, label='terminal e', ls='--', color='k')
    # ax[1,2].plot(taus, error_approx_halftime, label='halftime e', ls='-.', color='y')

    # ax[1,2].scatter((np.abs(det_temps-det_aggs) + np.abs(det_temps_halftime-det_aggs_halftime))*(2*tau/beta),
    #                 error_approx_combo, s=4, label='determ v combo', color=type_colors['algo'])
    ax[1,2].scatter(taus, (np.abs(det_temps-det_aggs) + np.abs(det_temps_halftime-det_aggs_halftime))*(2*tau/beta),
                     s=4, label='determ', color='y')
    ax[1,2].set_xlabel('Tau')
    ax[1,2].set_ylabel('Combination error X duration')
    ax[1,2].scatter(taus,
                    error_approx_combo, s=4, label='combo', color=type_colors['algo'])
    ax[1,2].legend(frameon=False)

    fig2, ax2 = plt.subplots()
    ax2.scatter((np.abs(det_temps-det_aggs) + np.abs(det_temps_halftime-det_aggs_halftime))*(2*tau/beta), error_approx_combo)

    y = np.linspace(0, max(np.abs(det_temps-det_aggs)), 10)
    ax[1, 1].plot(y, y, color='k', ls='--', alpha=0.6)
    y = np.linspace(0, max(np.abs(det_temps_halftime-det_aggs_halftime)), 10)
    ax[1, 0].plot(y, y, color='blue', ls='--', alpha=0.6)
    ax[1, 0].legend()
    ax[1, 1].legend()
    # plt.show()
    fig.set_size_inches((10,5))

    plt.tight_layout()
    plt.show()

N = 100
G3, A3 = configuration_model_graph(N)
G6, A6 = erdos_renyi_graph(N, .01)

A = A3
B = A6

beta = .12
taus = np.linspace(0.0001, 0.7, 50) # .7 for example

validation_on(A, B, beta, taus)

G3, A3 = cycle_graph(N)
G6, A6 = erdos_renyi_graph(N, .01)
G3, A3 = erdos_renyi_graph(N, .02)

A = A3
B = A6

beta = .12
taus = np.linspace(0.0001, 1.0, 50) # .7 for example

validation_on(A, B, beta, taus)
