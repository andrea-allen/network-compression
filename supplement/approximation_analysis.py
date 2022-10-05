import networkx as nx
import numpy as np
from scipy.linalg import expm
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import matplotlib
# plt.rcParams
# matplotlib.rcParams.update({'font.size': 12})

# Showing what a "small" beta X delta t is, basically should show what happens on some network
# when beta X delta t is too big and then the approximation of the error term doesn't match with the deterministic result
# there's not a clear calculation based on the network size for when this happens...

# Work through supplement material (derivation) to identify what derivation steps would be helpful to generate
# extra figures for to explain them better... (basically that way have a supplemental derivation in pictures)

# 1. First: figure of full computed deterministic solution vs. matrix exponential approximation
# 1a. Maybe can have figures for multiple networks, then plotted as a function of tau? Or should it be a function of t?
# 1b. Different networks to show solutions on different networks
# The whole plot could be X-axis: network density, Y-axis: beta X delta t, each plot is the two solutions as a function of t

# for values tau in 0, T run a deterministic temporal
def plot_matexp_vs_solution(A, beta, tau, N, ax):
    snapshot = Snapshot(0, 2* tau / beta, beta, A)
    y_init = snapshot.dd_normalized
    temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                 networks={2 * tau / beta: A})
    solution_t_temporal, solution_p = temp_model.solve_model()
    temporal_timeseries = np.sum(solution_p, axis=0)

    # matrix exponential approximation
    t_vals = np.linspace(0, 2 * tau / beta, 50)
    appx = np.zeros(len(t_vals))
    for i, t in enumerate(t_vals):
        i_t = np.sum(expm(beta * t * A).dot(y_init))
        appx[i] = i_t

    # plotting
    ax.plot(solution_t_temporal, temporal_timeseries, label='Deterministic solution', color='k')
    ax.plot(t_vals, appx, label='Matrix exponential approximation', ls='-.')
    ax.plot(t_vals, np.full(len(t_vals), N), ls=':', color='grey', label='Size of network')
    ax.set_ylim([0, N+20])
    ax.set_ylabel('Infected nodes')
    ax.set_xlabel('Time $t$')

def plot_approx_vs_solution(A, beta, tau, N, ax):
    # Plot the deterministic solution next to the solution of the exponential approximation (by taking
    # the Taylor series expansion to third order) just to motivate the approach, not that we end up using this
    # approximation by itself
    # In the supplement: Actually write out the Taylor series expansion
    snapshot = Snapshot(0, 2* tau / beta, beta, A)
    y_init = snapshot.dd_normalized
    temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                 networks={2 * tau / beta: A})
    solution_t_temporal, solution_p = temp_model.solve_model()
    temporal_timeseries = np.sum(solution_p, axis=0)

    # matrix exponential approximation
    t_vals = np.linspace(0, 2 * tau / beta, 50)
    appx = np.zeros(len(t_vals))
    appx_2nd = np.zeros(len(t_vals))
    for i, t in enumerate(t_vals):
        # i_t = I + A + A@A / 2 + A@A@A / 6
        A_scaled = beta * t * A
        taylor_3rd = np.identity(len(A)) + A_scaled + (A_scaled @ A_scaled)/2 + (A_scaled @ A_scaled @ A_scaled)/6
        i_t = np.sum(taylor_3rd.dot(y_init))
        appx[i] = i_t

        taylor_2nd = np.identity(len(A)) + A_scaled + (A_scaled @ A_scaled)/2
        i_t = np.sum(taylor_2nd.dot(y_init))
        appx_2nd[i] = i_t

    # plotting
    # ax.plot(solution_t_temporal, temporal_timeseries, label='fully temporal', color='k')
    ax.plot(t_vals, appx, label='3rd order approximation', ls='--')
    # ax.plot(t_vals, appx_2nd, label='2nd order approximation', ls='-.')
    ax.plot(t_vals, np.full(len(t_vals), N), ls=':', color='grey')
    ax.set_ylim([0, N+20])
    ax.set_ylabel('Infected nodes')
    ax.set_xlabel('Time $t$')

def plot_point_error_vs_solution(A, B, beta, taus, N, ax, ax2):
    # Redo this whole thing, as a loop over taus, kind of like in the proof of concept figure 2 code
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
    integral_solutions = np.zeros(len(taus))

    for t, tau in enumerate(taus):
        A_lay = Snapshot(0, tau / beta, beta, A)
        B_lay = Snapshot(tau / beta, 2 * tau / beta, beta, B)
        epsilon_terminal = Compressor.epsilon(A_lay, B_lay, error_type='terminal')
        epsilon_halftime = Compressor.epsilon(A_lay, B_lay, error_type='halftime')
        epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined')
        error_approx_terminal[t] = epsilon_terminal
        error_approx_halftime[t] = epsilon_halftime
        error_approx_combo[t] = epsilon_combo
        # for values tau in 0, T run a deterministic temporal
        y_init = A_lay.dd_normalized
        temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                     networks={tau / beta: A, 2 * tau / beta: B})
        solution_t_temporal, solution_p = temp_model.solve_model()
        temporal_timeseries = np.sum(solution_p, axis=0)
        final_temp = temporal_timeseries[-1]
        det_temps[t] = final_temp
        det_temps_halftime[t] = temporal_timeseries[int(len(temporal_timeseries) / 2)]
        # plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
        model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                networks={2 * tau / beta: (A + B) / 2})
        solution_t_agg, solution_p = model.solve_model()
        aggregate_timeseries = np.sum(solution_p, axis=0)
        # plt.show()
        final_agg = aggregate_timeseries[-1]
        det_aggs[t] = final_agg
        det_aggs_halftime[t] = aggregate_timeseries[int(len(aggregate_timeseries) / 2)]

    ax.plot(taus, error_approx_combo / (2 * taus / beta), ls='-.', lw=2,
                  label='prediction, $\\epsilon_{MID}+\\epsilon_{END}$')
    ax.plot(taus, (np.abs(det_temps - det_aggs) + np.abs(det_temps_halftime - det_aggs_halftime)), ls='-',
                  lw=2, color='k', alpha=0.6, label='true solution, $\\epsilon_{MID}+\\epsilon_{END}$')
    # ax.plot(taus, ((det_temps - det_aggs) + (det_temps_halftime - det_aggs_halftime)), ls='-',
    #         lw=2, color='k', alpha=0.6, label='true solution, $\\epsilon_{MID}+\\epsilon_{END}$')
    ax.set_xlabel('$\\beta \\cdot \\delta t$')
    ax.set_ylabel('Infected nodes')
    ax.legend(frameon=False, loc='upper left')

    midpoint = int(len(aggregate_timeseries) / 2)
    # ax2.plot(solution_t_temporal[:midpoint], temporal_timeseries[:midpoint], lw=1) #'m'
    # ax2.plot(solution_t_temporal[midpoint:], temporal_timeseries[midpoint:],  lw=1)
    # ax2.plot(solution_t_agg, aggregate_timeseries, ls='--', lw=1)
    # ax2.vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
    #                 ymax=temporal_timeseries[-1], color='k', lw=1)
    # ax2.vlines(tau / beta,
    #                 ymin=aggregate_timeseries[midpoint],
    #                 ymax=temporal_timeseries[midpoint],
    #                 color='k', lw=1)
    # ax2.fill_between(solution_t_temporal, aggregate_timeseries, temporal_timeseries,
    #                       alpha=0.25)
    # # ax2.text(tau / beta - 4, temporal_timeseries[midpoint] + 3, 'temporal solution')
    # # ax2.text(tau / beta, aggregate_timeseries[midpoint] - 2, 'aggregate solution')
    # # ax2.text(tau / beta, temporal_timeseries[midpoint] - .2*temporal_timeseries[midpoint], '$\\epsilon_{MID}$')
    # # ax2.text(2*tau / beta, temporal_timeseries[-1] - .1*temporal_timeseries[-1], '$\\epsilon_{END}$')

def plot_integrated_error_vs_xi(A, B, beta, taus, N, ax, ax2):
    # Redo this whole thing, as a loop over taus, kind of like in the proof of concept figure 2 code
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
    integral_solutions = np.zeros(len(taus))

    for t, tau in enumerate(taus):
        A_lay = Snapshot(0, tau / beta, beta, A)
        B_lay = Snapshot(tau / beta, 2 * tau / beta, beta, B)
        epsilon_terminal = Compressor.epsilon(A_lay, B_lay, error_type='terminal')
        epsilon_halftime = Compressor.epsilon(A_lay, B_lay, error_type='halftime')
        epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined')
        error_approx_terminal[t] = epsilon_terminal
        error_approx_halftime[t] = epsilon_halftime
        error_approx_combo[t] = epsilon_combo
        # for values tau in 0, T run a deterministic temporal
        y_init = A_lay.dd_normalized
        temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                     networks={tau / beta: A, 2 * tau / beta: B})
        solution_t_temporal, solution_p = temp_model.solve_model()
        temporal_timeseries = np.sum(solution_p, axis=0)
        final_temp = temporal_timeseries[-1]
        det_temps[t] = final_temp
        det_temps_halftime[t] = temporal_timeseries[int(len(temporal_timeseries) / 2)]
        # plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
        model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                networks={2 * tau / beta: (A + B) / 2})
        solution_t_agg, solution_p = model.solve_model()
        aggregate_timeseries = np.sum(solution_p, axis=0)
        # plt.show()
        final_agg = aggregate_timeseries[-1]
        det_aggs[t] = final_agg
        det_aggs_halftime[t] = aggregate_timeseries[int(len(aggregate_timeseries) / 2)]

        integrate_between = integrate_error_ts(temporal_ts=solution_t_temporal, temporal_inf=temporal_timeseries,
                                               other_ts=solution_t_agg, other_inf=aggregate_timeseries)
        integral_solutions[t] = integrate_between

    ax.scatter(integral_solutions,error_approx_combo,
                    marker='s', alpha=0.3,
                    label='increasing $\\beta\\cdot\\delta t$')
    ax.set_xlabel('True solution integrated error')
    ax.set_ylabel('Error measure $\\xi_{A,B}$')
    ax.legend(frameon=False, loc='upper left')

    midpoint = int(len(aggregate_timeseries) / 2)
    # ax2.plot(solution_t_temporal[:midpoint], temporal_timeseries[:midpoint], lw=1) #'m'
    # ax2.plot(solution_t_temporal[midpoint:], temporal_timeseries[midpoint:],  lw=1)
    # ax2.plot(solution_t_agg, aggregate_timeseries, ls='--', lw=1)
    # ax2.vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
    #                 ymax=temporal_timeseries[-1], color='k', lw=1)
    # ax2.vlines(tau / beta,
    #                 ymin=aggregate_timeseries[midpoint],
    #                 ymax=temporal_timeseries[midpoint],
    #                 color='k', lw=1)
    # ax2.fill_between(solution_t_temporal, aggregate_timeseries, temporal_timeseries,
    #                       alpha=0.25)
    #
    # ### Filling between for the approximation
    # temporal_mid = np.full( len(solution_t_temporal), temporal_timeseries[midpoint])
    # temporal_end = np.full( len(solution_t_temporal), temporal_timeseries[-1])
    # aggregate_mid = np.full(len(solution_t_temporal), aggregate_timeseries[midpoint])
    # aggregate_end = np.full(len(solution_t_temporal), aggregate_timeseries[-1])
    # ax2.fill_between(solution_t_temporal, temporal_mid, aggregate_mid,
    #                       alpha=0.25)
    # ax2.fill_between(solution_t_temporal, temporal_end, aggregate_end,
    #                       alpha=0.25)
    # # ax2.text(tau / beta - 4, temporal_timeseries[midpoint] + 3, 'temporal solution')
    # # ax2.text(tau / beta, aggregate_timeseries[midpoint] - 2, 'aggregate solution')
    # # ax2.text(tau / beta, temporal_timeseries[midpoint] - .2*temporal_timeseries[midpoint], '$\\epsilon_{MID}$')
    # # ax2.text(2*tau / beta, temporal_timeseries[-1] - .1*temporal_timeseries[-1], '$\\epsilon_{END}$')



sns.set_palette('Set1')
# Work in progress, this idea is good so far though
N = 100
G3, A3 = configuration_model_graph(N)
G6, A6 = erdos_renyi_graph(N, .02)
G7, A7 = erdos_renyi_graph(N, .05)
G8, A8 = erdos_renyi_graph(N, .08)
_, axs1 = plt.subplots(1, 3, sharey=True)
axs1[0].set_title('Heterogeneous\nnetwork')
axs1[1].set_title('ER network with\naverage degree 5')
axs1[2].set_title('ER network with\naverage degree 8')

plot_matexp_vs_solution(A3, beta=.1, tau=.4, N=N, ax=axs1[0])
# plot_matexp_vs_solution(A6, beta=.1, tau=.7, N=N, ax=axs1[1])
plot_matexp_vs_solution(A7, beta=.1, tau=.7, N=N, ax=axs1[1])
plot_matexp_vs_solution(A8, beta=.1, tau=.7, N=N, ax=axs1[2])

plot_approx_vs_solution(A3, beta=.1, tau=.4, N=N, ax=axs1[0])
# plot_approx_vs_solution(A6, beta=.1, tau=.7, N=N, ax=axs1[1])
plot_approx_vs_solution(A7, beta=.1, tau=.7, N=N, ax=axs1[1])
plot_approx_vs_solution(A8, beta=.1, tau=.7, N=N, ax=axs1[2])

axs1[0].legend(loc='upper left',frameon=False)
plt.show()

# TODO Forgot to show a good supplementary in between figure which is not only the full matrix exponential
# but also the third order approximation compared with the deterministic solution
# maybe also compared with the full matrix exponential? All for one network at a time but then you can
# see it controlling the approximation
# discuss what it means *mechanistically*, as in it's only using up to order 3 paths



# 2. Same plotting idea as #1, but with the aggregate and temporal approximations of the BCH derived thing vs the
# deterministic solution (like in paper figure)
# Each subplot, instead of a single network, will have to be two unique networks for the temporal experiment
# Next: plot of end point and mid point actual error, vs deterministic "final" error at end and midpoint,
# Stress that mathematically it doesn't matter if its beta or tau
_, axs = plt.subplots(5, 2)
plot_point_error_vs_solution(A3, A6, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[0, 0], ax2=axs[0, 1])
plot_point_error_vs_solution(A3, A7, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[1, 0], ax2=axs[1, 1])
plot_point_error_vs_solution(A3, A8, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[2, 0], ax2=axs[2, 1])
plot_point_error_vs_solution(A6, A8, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[3, 0], ax2=axs[3, 1])
plot_point_error_vs_solution(A7, A8, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[4, 0], ax2=axs[4, 1])


# 3. Final comparison of same plot but with the integral, and the time-penalty approximation (xi term)
# Then figures 1-3 will show the build-up of the derivation for why it makes sense and holds up

# _, axs3 = plt.subplots(5, 2)
plot_integrated_error_vs_xi(A3, A6, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[0, 1], ax2=axs[0, 1])
plot_integrated_error_vs_xi(A3, A7, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[1, 1], ax2=axs[1, 1])
plot_integrated_error_vs_xi(A3, A8, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[2, 1], ax2=axs[2, 1])
plot_integrated_error_vs_xi(A6, A8, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[3, 1], ax2=axs[3, 1])
plot_integrated_error_vs_xi(A7, A8, beta=.1, taus=np.linspace(0.0001, .4, 20), N=N, ax=axs[4, 1], ax2=axs[4, 1])

axs[0, 0].set_ylim([0, 15])
axs[1, 0].set_ylim([0, 15])
axs[2, 0].set_ylim([0, 15])
axs[3, 0].set_ylim([0, 15])
axs[4, 0].set_ylim([0, 15])
axs[0, 1].set_ylim([0, 100])
axs[1, 1].set_ylim([0, 100])
axs[2, 1].set_ylim([0, 100])
axs[3, 1].set_ylim([0, 100])
axs[4, 1].set_ylim([0, 100])


plt.show()



# TODO Would be good to also do a plot that shows that monotonicity argument
# 4. Time penalty score visualization / explanation (there is a proof of concept for this that I coded up, that
# the boxes get bigger montonically)