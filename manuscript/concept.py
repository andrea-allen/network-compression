import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.linalg import expm
from manuscript.simulation_helpers import *
from manuscript.network_generators import *
from manuscript.plotters import *
import matplotlib
plt.rcParams
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


cmr_map = sns.color_palette('CMRmap_r', 6)
type_colors = {'temp': 'grey', 'even': cmr_map[1], 'algo': cmr_map[5],
               'snap1': cmr_map[2], 'snap2': cmr_map[3]}
type_colors = {'temp': 'grey', 'even': 'mediumblue', 'algo': 'darkorange',
               'snap1': cmr_map[2], 'snap2': cmr_map[3]}
print(type_colors)


def multi_panel_fig_idea2(snapshots, beta, increments):
    # fig, ax = plt.subplots(1, len(increments)-1, sharey=True, sharex=True)
    fig, ax = plt.subplots()
    final_temporal_y_points = [0]
    colors = ['m', 'c', 'gold', 'firebrick', 'indigo', 'forestgreen']
    start_x = 0
    for i in range(len(increments)-1):
        tA = increments[i]
        dA = increments[i] - start_x
        tB = increments[i+1]
        dB = increments[i+1] - tA
        A = snapshots[i]
        B = snapshots[i+1]
        A_lay = Snapshot(0, dA, beta, snapshots[i])
        B_lay = Snapshot(dA, dA+dB, beta, snapshots[i+1])
        epsilon_terminal = Compressor.epsilon(A_lay, B_lay, error_type='terminal')
        epsilon_halftime = Compressor.epsilon(A_lay, B_lay, error_type='halftime')
        epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined')
        # for values tau in 0, T run a deterministic temporal
        y_init = A_lay.dd_normalized
        temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=dA+dB,
                                     networks={dA: A, dA+dB: B})
        solution_t_temporal, solution_p = temp_model.solve_model(custom_t_inc=.01)
        temporal_timeseries = np.sum(solution_p, axis=0)
        final_temp = temporal_timeseries[-1]
        final_temporal_y_points.append(final_temp)
        model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=dA+dB,
                                networks={dA+dB: (dA*A + dB*B)/(dA+dB)})
        solution_t_agg, solution_p = model.solve_model(custom_t_inc=.01)
        aggregate_timeseries = np.sum(solution_p, axis=0)
        # plt.show()
        final_agg = aggregate_timeseries[-1]

        midpoint_a = 0
        for t in range(len(solution_t_agg)):
            if solution_t_agg[midpoint_a] < dA:
                if solution_t_agg[t] > dA:
                    midpoint_a = t
        midpoint_t = 0
        for t in range(len(solution_t_temporal)):
            if solution_t_agg[midpoint_t] < dA:
                if solution_t_temporal[t] > dA:
                    midpoint_t = t
        # midpoint_t = np.where(np.array(solution_t_temporal) == increments[i]+.01)[0][0]
        ax.plot(start_x + np.array(solution_t_temporal)[:midpoint_t],  temporal_timeseries[:midpoint_t], color=colors[i], lw=1, label=f'Snapshot {i+1}')
        if i+1 == len(increments) -1:
            ax.plot(start_x + np.array(solution_t_temporal)[midpoint_t:],  temporal_timeseries[midpoint_t:], color=colors[i+1], lw=1, label=f'Snapshot {i+2}')
        else:
            ax.plot(start_x + np.array(solution_t_temporal)[midpoint_t:],  temporal_timeseries[midpoint_t:], color=colors[i+1], lw=1)
        ax.plot(start_x + np.array(solution_t_agg),  aggregate_timeseries, color='grey', ls='--', lw=1)
        ax.vlines(start_x + np.array(solution_t_temporal)[-1], ymin= aggregate_timeseries[-1],
                        ymax= temporal_timeseries[-1], color='crimson', lw=1)
        ax.vlines(start_x + dA,
                        ymin= aggregate_timeseries[midpoint_a],
                        ymax= temporal_timeseries[midpoint_t],
                        color='crimson', lw=1)
        ax.fill_between(start_x + np.array(solution_t_agg),  aggregate_timeseries,  temporal_timeseries, color='grey',
                              alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        start_x = increments[i]
    ax.legend(frameon=False, loc='upper left')


def multi_panel_fig_idea(snapshots, beta, increments):
    S1 = Snapshot(0, increments[0], beta, snapshots[0])
    y_init = S1.dd_normalized
    temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=increments[-1],
                                 networks={increments[i]: snapshots[i] for i in range(len(increments))})
    solution_t_temporal, solution_p, p_states = temp_model.solve_model(return_p_vecs=True)
    temporal_timeseries = np.sum(solution_p, axis=0)

    colors = ['m', 'c', 'gold', 'lime', 'orange', 'purple', 'blue', 'indigo']
    fig, ax = plt.subplots(1,2)
    fig2, ax2 = plt.subplots(1,5, sharey=True, sharex=False)
    start = 0
    temporal_midpoint_keys = []
    for i in range(len(increments)):
        midpoint_t = np.where(np.array(solution_t_temporal) == increments[i])[0][0]
        temporal_midpoint_keys.append(midpoint_t)
        ax[0].plot(solution_t_temporal[start:midpoint_t], temporal_timeseries[start:midpoint_t], color=colors[i], lw=1)
        if i < len(increments)-1:
            ax2[i].plot(solution_t_temporal[start:midpoint_t], temporal_timeseries[start:midpoint_t], color=colors[i], lw=1)
        if i > 0:
            ax2[i-1].plot(solution_t_temporal[start:midpoint_t], temporal_timeseries[start:midpoint_t], color=colors[i], lw=1)
        start = midpoint_t

    snap_1_duration = increments[0]

    new_y_init = y_init
    start_y = 0
    start_x = 0
    orig_start = 0
    for i in range(len(increments)):
        try:
            snap_2_duration = increments[i+1] - increments[i]
            y_init = p_states[i]
            # y_init = Snapshot(orig_start, increments[i], beta, snapshots[i]).dd_normalized
            agg_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=snap_1_duration+snap_2_duration,
                                networks={snap_1_duration+snap_2_duration: (snap_1_duration*snapshots[i] + snap_2_duration*snapshots[i+1]) / (snap_2_duration+snap_1_duration)})
            epsilon_combo = Compressor.epsilon(Snapshot(orig_start, increments[i], beta, snapshots[i]), Snapshot(increments[i], increments[i+1], beta, snapshots[i+1]), error_type='combined')
            orig_start = increments[i]
            solution_t_agg, solution_p = agg_model.solve_model()
            aggregate_timeseries = np.sum(solution_p, axis=0)
            new_y_init = solution_p[:,-1] # needs to be the y init of the temporal solution
            midpoint_a = 0
            for t in range(len(solution_t_agg)):
                if solution_t_agg[midpoint_a] < snap_1_duration:
                    if solution_t_agg[t] > snap_1_duration:
                        midpoint_a = t
            try:
                ax[0].plot(start_x + np.array(solution_t_agg), np.array(aggregate_timeseries), color='grey', ls='--', lw=1)
                ax2[i].plot(start_x + np.array(solution_t_agg), np.array(aggregate_timeseries), color='grey', ls='--', lw=1)
                ax[0].vlines(increments[i],
                              ymin=aggregate_timeseries[midpoint_a],
                              ymax=temporal_timeseries[temporal_midpoint_keys[i]],
                              color='crimson', lw=1)
                ax2[i].vlines(increments[i],
                             ymin=aggregate_timeseries[midpoint_a],
                             ymax=temporal_timeseries[temporal_midpoint_keys[i]],
                             color='crimson', lw=1)
                ax[0].fill_between(start_x + np.array(solution_t_agg),
                                    np.full(len(solution_t_agg), aggregate_timeseries[midpoint_a]),
                                    np.full(len(solution_t_agg), temporal_timeseries[temporal_midpoint_keys[i]]),
                                    color=colors[i], alpha=0.35, label=np.round(
                        np.abs(aggregate_timeseries[midpoint_a] - temporal_timeseries[temporal_midpoint_keys[i]]) * (snap_1_duration+snap_2_duration), 2))
                ax2[i].fill_between(start_x + np.array(solution_t_agg),
                                   np.full(len(solution_t_agg), aggregate_timeseries[midpoint_a]),
                                   np.full(len(solution_t_agg), temporal_timeseries[temporal_midpoint_keys[i]]),
                                   color=colors[i], alpha=0.35, label=np.round(
                        np.abs(aggregate_timeseries[midpoint_a] - temporal_timeseries[temporal_midpoint_keys[i]]) * (
                                    snap_1_duration + snap_2_duration), 2))
                ax2[i].fill_between(start_x + np.array(solution_t_agg),
                                   aggregate_timeseries,
                                   temporal_timeseries[temporal_midpoint_keys[i-1]: temporal_midpoint_keys[i+1]],
                                   color=colors[i], alpha=0.35, label=np.round(
                        np.abs(aggregate_timeseries[midpoint_a] - temporal_timeseries[temporal_midpoint_keys[i]]) * (
                                    snap_1_duration + snap_2_duration), 2))
                # hatch='///', zorder=2, fc='c',)
                endpoint_t = np.where(np.array(solution_t_temporal) == increments[i + 1])[0][0]
                ax[0].fill_between(start_x + np.array(solution_t_agg),
                                    np.full(len(solution_t_agg), aggregate_timeseries[-1]),
                                    np.full(len(solution_t_agg), temporal_timeseries[endpoint_t]),
                                    color=colors[i], alpha=0.35,
                                    label=np.round(np.abs(aggregate_timeseries[-1] - temporal_timeseries[endpoint_t]) * (snap_1_duration+snap_2_duration),
                                                   2))
                ax2[i].fill_between(start_x + np.array(solution_t_agg),
                                    np.full(len(solution_t_agg), aggregate_timeseries[-1]),
                                    np.full(len(solution_t_agg), temporal_timeseries[endpoint_t]),
                                    color=colors[i], alpha=0.35,
                                    label=np.round(np.abs(aggregate_timeseries[-1] - temporal_timeseries[endpoint_t]) * (snap_1_duration+snap_2_duration),
                                                   2))
                ax[1].fill_between(start_x + np.array(solution_t_agg),
                                    np.full(len(solution_t_agg), aggregate_timeseries[midpoint_a]),
                                    np.full(len(solution_t_agg),
                                            aggregate_timeseries[midpoint_a] + epsilon_combo / (snap_1_duration+snap_2_duration)),
                                    color=colors[i], alpha=0.4, label=f'Approximation:{np.round(epsilon_combo, 2)}')
                start_x = increments[i]
                ax[0].legend(frameon=False)
                ax[1].legend(frameon=False)
                ax2[i].spines['right'].set_visible(False)
                ax2[i].spines['top'].set_visible(False)
                ax2[i].spines['left'].set_visible(False)
                ax2[i].spines['bottom'].set_visible(True)
                ax2[i].legend(frameon=False, loc='lower right')
                ax2[i].set_xlabel('t')
            except:
                ax[0].plot(solution_t_agg, aggregate_timeseries, color='grey', ls='--', lw=1)

        except:
            pass
        snap_1_duration = snap_2_duration
    ax2[0].set_ylabel('Infected nodes')


def concept_custom_durations(A, B, beta, tA, tB):
    A_lay = Snapshot(0, tA, beta, A)
    B_lay = Snapshot(tA, tB, beta, B)
    C_lay = Snapshot(tB, tB+tB, beta, A)
    # epsilon_terminal = Compressor.epsilon(A_lay, B_lay, error_type='terminal')
    # epsilon_halftime = Compressor.epsilon(A_lay, B_lay, error_type='halftime')
    epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined')
    # for values tau in 0, T run a deterministic temporal
    y_init = A_lay.dd_normalized
    temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=tB+tB,
                                 networks={tA: A, tB: B, tB+tB: A})
    solution_t_temporal, solution_p = temp_model.solve_model()
    temporal_timeseries = np.sum(solution_p, axis=0)
    final_temp = temporal_timeseries[-1]
    model_agg1 = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=tB,
                            networks={tB: (tA*A + (tB-tA)*B) / (tB), tB+tB: A})
    model_agg2 = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=tB,
                            networks={tA: A, tB+tB: ((tB-tA)*B + (tB)*A) / ((tB+tB))})
    solution_t_agg, solution_p = model_agg1.solve_model()
    aggregate_timeseries = np.sum(solution_p, axis=0)
    # plt.show()
    final_agg = aggregate_timeseries[-1]
    P0 = A_lay.dd_normalized
    # matexp_temp = np.sum(expm(tau * B).dot(expm(tau * A).dot(P0)))
    # matexp_agg = np.sum(expm(2 * tau * (B + A) / 2).dot(P0))

    fig2, ax2 = plt.subplots(1,2, sharey=True, sharex=True)
    midpoint_t = np.where(np.array(solution_t_temporal)==tA)[0][0]
    midpoint_a = 0
    for t in range(len(solution_t_agg)):
        if solution_t_agg[midpoint_a] < tA:
            if solution_t_agg[t] > tA:
                midpoint_a = t
    print(midpoint_a)
    print(solution_t_agg[midpoint_a])
    # midpoint_a = np.where(np.array(solution_t_agg)==12)[0][0]

    ax2[0].plot(solution_t_temporal[:midpoint_t], temporal_timeseries[:midpoint_t], color='m', lw=1)
    ax2[0].plot(solution_t_temporal[midpoint_t:], temporal_timeseries[midpoint_t:], color='c', lw=1)
    ax2[0].plot(solution_t_agg, aggregate_timeseries, color='grey', ls='--', lw=1)
    ax2[0].vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
                    ymax=temporal_timeseries[-1], color='crimson', lw=1)
    ax2[0].vlines(tA,
                    ymin=aggregate_timeseries[midpoint_a],
                    ymax=temporal_timeseries[midpoint_t],
                    color='crimson', lw=1)
    ax2[0].fill_between(solution_t_temporal,
                     np.full(len(solution_t_temporal), aggregate_timeseries[midpoint_a]),
                     np.full(len(solution_t_temporal), temporal_timeseries[midpoint_t]),
                     color='gold', alpha=0.5, label=np.round(np.abs(aggregate_timeseries[midpoint_a]-temporal_timeseries[midpoint_t])*(tB), 2))
                    # hatch='///', zorder=2, fc='c',)
    ax2[0].fill_between(solution_t_temporal,
                     np.full(len(solution_t_temporal), aggregate_timeseries[-1]),
                     np.full(len(solution_t_temporal), temporal_timeseries[-1]),
                     color='gold', alpha=0.5, label=np.round(np.abs(aggregate_timeseries[-1]-temporal_timeseries[-1])*(tB),2))
                     # hatch='///', zorder=2, fc='c',)

    ax2[1].plot(solution_t_temporal[:midpoint_t], temporal_timeseries[:midpoint_t], color='m', lw=1)
    ax2[1].plot(solution_t_temporal[midpoint_t:], temporal_timeseries[midpoint_t:], color='c', lw=1)
    ax2[1].plot(solution_t_agg, aggregate_timeseries, color='grey', ls='--', lw=1)
    ax2[1].vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
                    ymax=temporal_timeseries[-1], color='crimson', lw=1)
    ax2[1].vlines(tA,
                    ymin=aggregate_timeseries[midpoint_a],
                    ymax=temporal_timeseries[midpoint_t],
                    color='crimson', lw=1)
    ax2[1].fill_between(solution_t_temporal,
                     np.full(len(solution_t_temporal), aggregate_timeseries[midpoint_a]),
                     np.full(len(solution_t_temporal), aggregate_timeseries[midpoint_a] + epsilon_combo / (tB)),
                     color='m', alpha=0.4, label=f'Approximation:{np.round(epsilon_combo,2)}')
    ax2[0].spines['right'].set_visible(False)
    ax2[0].spines['top'].set_visible(False)
    ax2[1].spines['right'].set_visible(False)
    ax2[1].spines['top'].set_visible(False)
    ax2[0].legend(frameon=False, loc='lower right')
    ax2[1].legend(frameon=False, loc='lower right')


"""
Fig. 2 in paper
"""
def manuscript_fig2(A, B, beta, taus):
    error_approx_terminal = np.zeros(len(taus))
    error_approx_halftime = np.zeros(len(taus))
    error_approx_combo = np.zeros(len(taus))
    error_approx_o3 = np.zeros(len(taus))
    matexp_temps = np.zeros(len(taus))
    matexp_temps_h = np.zeros(len(taus))
    matexp_aggs = np.zeros(len(taus))
    matexp_aggs_h = np.zeros(len(taus))
    det_temps = np.zeros(len(taus))
    det_temps_halftime = np.zeros(len(taus))
    det_aggs = np.zeros(len(taus))
    det_aggs_halftime = np.zeros(len(taus))
    integral_solutions = np.zeros(len(taus))

    # type_colors = {'temp': 'grey', 'even': "#FFC626", 'algo': "#00A4D4"}
    tau_color = sns.color_palette('Greys', len(taus))

    fig, ax = plt.subplots(2, 2, figsize=(6,6))
    for t, tau in enumerate(taus):
        A_lay = Snapshot(0, tau / beta, beta, A)
        B_lay = Snapshot(tau / beta, 2 * tau / beta, beta, B)
        epsilon_terminal = Compressor.epsilon(A_lay, B_lay, error_type='terminal')
        epsilon_halftime = Compressor.epsilon(A_lay, B_lay, error_type='halftime')
        epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined', order=1, norm=2)
        epsilon_combo_o3 = Compressor.epsilon(A_lay, B_lay, error_type='combined', order=3, norm=2)
        error_approx_terminal[t] = epsilon_terminal
        error_approx_halftime[t] = epsilon_halftime
        error_approx_combo[t] = epsilon_combo
        error_approx_o3[t] = epsilon_combo_o3
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
        # det_temp = _
        # det_agg = _
        # For values tau in 0, T run the matrix approximation temporal
        P0 = np.full(N, 1 / N)
        P0 = A_lay.dd_normalized
        matexp_temp = np.sum(expm(tau * B).dot(expm(tau * A).dot(P0)))
        matexp_agg = np.sum(expm(2 * tau * (B + A) / 2).dot(P0))
        matexp_temps[t] = matexp_temp
        matexp_aggs[t] = matexp_agg
        matexp_temps_h[t] = np.sum(expm(tau * A).dot(P0))
        matexp_aggs_h[t] = np.sum(expm(tau * (B + A) / 2).dot(P0))
        integrate_between = integrate_error_ts(temporal_ts=solution_t_temporal, temporal_inf=temporal_timeseries,
                                               other_ts=solution_t_agg, other_inf=aggregate_timeseries)
        integral_solutions[t] = integrate_between

    midpoint = int(len(aggregate_timeseries) / 2)
    ax[0, 1].plot(solution_t_temporal[:midpoint], temporal_timeseries[:midpoint], color=type_colors['snap1'], lw=1) #'m'
    ax[0, 1].plot(solution_t_temporal[midpoint:], temporal_timeseries[midpoint:], color=type_colors['snap2'], lw=1)
    ax[0, 1].plot(solution_t_agg, aggregate_timeseries, color=tau_color[t], ls='--', lw=1)
    ax[0, 1].vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
                    ymax=temporal_timeseries[-1], color='k', lw=1)
    ax[0, 1].vlines(tau / beta,
                    ymin=aggregate_timeseries[midpoint],
                    ymax=temporal_timeseries[midpoint],
                    color='k', lw=1)
    ax[0, 1].fill_between(solution_t_temporal, aggregate_timeseries, temporal_timeseries, color=tau_color[t],
                          alpha=0.25)
    ax[0, 1].text(tau / beta - 4.5, temporal_timeseries[midpoint] + 3, 'temporal solution')
    ax[0, 1].text(tau / beta, aggregate_timeseries[midpoint] - 2.5, 'aggregate solution') # TODO move these down

    # ax[0, 1].text(tau / beta + .05*(tau / beta), temporal_timeseries[midpoint] - .2*temporal_timeseries[midpoint], '$I(t_1^A)$')
    # ax[0, 1].text(tau / beta + 0.4*(tau / beta), temporal_timeseries[-1] - .1*temporal_timeseries[-1], '$I(t_1^B)$') #TODO take these away



    ### Filling between for the approximation
    temporal_mid = np.full( len(solution_t_temporal), temporal_timeseries[midpoint])
    temporal_end = np.full( len(solution_t_temporal), temporal_timeseries[-1])
    aggregate_mid = np.full(len(solution_t_temporal), aggregate_timeseries[midpoint])
    aggregate_end = np.full(len(solution_t_temporal), aggregate_timeseries[-1])
    ax[0, 1].fill_between(solution_t_temporal, temporal_mid, aggregate_mid, color=type_colors['algo'],
                          alpha=0.25)
    ax[0, 1].fill_between(solution_t_temporal, temporal_end, aggregate_end, color=type_colors['algo'],
                          alpha=0.25)


    ####

    # fig2, ax2 = plt.subplots()
    # midpoint = int(len(aggregate_timeseries) / 2)
    # ax2.plot(solution_t_temporal[:midpoint], temporal_timeseries[:midpoint], color=type_colors['snap1'], lw=1)
    # ax2.plot(solution_t_temporal[midpoint:], temporal_timeseries[midpoint:], color=type_colors['snap2'], lw=1)
    # ax2.plot(solution_t_agg, aggregate_timeseries, color=tau_color[t], ls='--', lw=1)
    # ax2.vlines(solution_t_temporal[-1], ymin=aggregate_timeseries[-1],
    #                 ymax=temporal_timeseries[-1], color='crimson', lw=1)
    # ax2.vlines(tau / beta,
    #                 ymin=aggregate_timeseries[midpoint],
    #                 ymax=temporal_timeseries[midpoint],
    #                 color='crimson', lw=1)
    # ax2.fill_between(solution_t_temporal[:midpoint],
    #                  np.full(len(solution_t_temporal[:midpoint]), aggregate_timeseries[midpoint]),
    #                  np.full(len(solution_t_temporal[:midpoint]), temporal_timeseries[midpoint]),
    #                  color='gold', alpha=0.5)
    #                 # hatch='///', zorder=2, fc='c',alpha=0.2)
    # ax2.fill_between(solution_t_temporal[midpoint:],
    #                  np.full(len(solution_t_temporal[midpoint:]), aggregate_timeseries[midpoint]),
    #                  np.full(len(solution_t_temporal[midpoint:]), temporal_timeseries[midpoint]),
    #                  color='gold', alpha=0.5)
    #                 # hatch='///', zorder=2, fc='c',)
    # ax2.fill_between(solution_t_temporal,
    #                  np.full(len(solution_t_temporal), aggregate_timeseries[-1]),
    #                  np.full(len(solution_t_temporal), temporal_timeseries[-1]),
    #                  color='green', alpha=0.5)
    #                  # hatch='///', zorder=2, fc='c',)

    # Degree distributions plots
    sns.distplot([sum(A[n]) for n in range(N)], label='snapshot 1', color=type_colors['snap1'], ax=ax[0,0], hist=False, kde_kws={'clip': (0.0, 20.0), 'bw':1.1})
    sns.distplot([sum(B[n]) for n in range(N)], label='snapshot 2', color=type_colors['snap2'], ax=ax[0,0], hist=False, kde_kws={'clip': (0.0, 20.0), 'bw':1.1})
    mean_snapshot1 = np.mean([sum(A[n]) for n in range(N)])
    mean_snapshot2 = np.mean([sum(B[n]) for n in range(N)])
    ax[0,0].axvline(mean_snapshot1, ls='--', color=type_colors['snap1'])
    ax[0,0].axvline(mean_snapshot2, ls='--', color=type_colors['snap2'])
    ax[0,0].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax[0,0].legend(frameon=False)
    ax[0,0].set_xlim([0,18])

    # fig3, ax3 = plt.subplots()
    ### ALT 0,1 AX
    # Integral between curves vs approximation error term, showing not equality or linearity but monotonicity

    ### 7/21 idea: Ranking, instead of scatter plots? Of argsort of true vs argsort of prediction
    ax[1,1].scatter(np.argsort(integral_solutions),np.argsort(error_approx_combo), color=type_colors['algo'],
                    marker='o', alpha=0.5)
    # ax[1,1].scatter(np.argsort(integral_solutions),np.argsort(error_approx_o3), color=cmr_map[0],
    #                 marker='d', alpha=0.5,
    #                 label='O(3)')
    # ax[1,1].scatter(taus,error_approx_combo/max(error_approx_combo), color=algo_blue,
    #                 marker='s', alpha=0.3,
    #                 label='error measure $\\xi_{A,B}$')
    # ax[1,1].scatter(taus,integral_solutions/max(integral_solutions), color='grey',
    #                 marker='s', alpha=0.3,
    #                 label='true solution integral')
    # ax[0,0].plot(taus, integral_solutions, ls='-', lw=2, color='k', label='true solution')
    ax[1,1].set_xlabel('True solution integrated error')
    ax[1,1].set_xlabel('Rank of area between solutions')
    # ax[1,1].set_xlabel('$\\beta \\cdot \\delta t$')
    ax[1,1].set_ylabel('Error measure $\\xi_{A,B}$')
    ax[1,1].set_ylabel('Rank of $\\xi_{A,B}$')
    # ax[1,1].set_ylabel('Infected nodes error')
    ax[1,1].legend(frameon=False, loc='upper left')
    #####
    # diff_matexp = matexp_temp - matexp_agg
    # diff_det = det_temp - det_agg
    # # Fig 2a: plot tau vs det_temp, det_agg, matexp_temp, matexp_agg
    # ax3.plot(taus, det_temps, label='ODE, temporal', alpha=0.9, color='black', ls='-')
    # ax3.plot(taus, det_aggs, label='ODE, aggregate', alpha=0.6, color='black', ls='--')
    # ax3.plot(taus, matexp_temps, label='$exp(\\beta\\delta t B)exp(\\beta\\delta t A)$', alpha=0.5, color='grey', ls='-')
    # ax3.plot(taus, matexp_aggs, label='$exp(\\beta(2\\delta t) \\overline{A+B})$', alpha=0.5, color='grey', ls='--')
    # ax3.set_xlabel('$\\beta \\delta t$')
    # ax3.set_ylabel('number nodes infected \nafter $2\\delta t$ time')
    # ax3.legend(frameon=False)

    ## SQUARE 1,1 option
    # ax[1,1].scatter((np.abs(det_temps - det_aggs) + np.abs(det_temps_halftime - det_aggs_halftime)),
    #             error_approx_combo / (2 * taus / beta), marker='o', fc='none', color=algo_blue)
    # ax[1,1].set_xlabel('ODE solution error')
    # ax[1,1].set_ylabel('Predicted error $\\xi$')
    # ax[1,1].plot(error_approx_combo/(2*taus/beta), error_approx_combo/(2*taus/beta), color='grey', ls=':')

    ## ALT SQUARE 1,1
    ax[1,0].plot(taus,error_approx_combo/ (2 * taus / beta), ls='-.', lw=2, color=type_colors['algo'], label='$\\epsilon_{MID}+\\epsilon_{END}$') #TODO: error emid and eend
    # ax[1,0].plot(taus,error_approx_o3/ (2 * taus / beta), ls='-.', lw=2, color=cmr_map[0], label='O(3), $\\epsilon_{MID}+\\epsilon_{END}$')
    # ax[1,0].plot(taus,  (np.abs(det_temps - det_aggs) + np.abs(det_temps_halftime - det_aggs_halftime)), ls='-',
    #              lw=2, color='k', alpha=0.6, label='$|I(t_1^B)_{TEMP} - I(t_1^B)_{AGG}|$\n'
    #                                                '$+|I(t_1^A)_{TEMP} - I(t_1^A)_{AGG}|$') #TODO: error between ODEs
    ax[1,0].plot(taus,  (np.abs(det_temps - det_aggs) + np.abs(det_temps_halftime - det_aggs_halftime)), ls='-',
                 lw=2, color='k', alpha=0.6, label='error between\nsolutions') #TODO: label as error between ODEs
    ax[1,0].set_xlabel('$\\beta \\cdot \\delta t$')
    ax[1,0].set_ylabel('Infected nodes')
    ax[1,0].legend(frameon=False, loc='upper left')

    #####

    # Labels
    ax[0, 1].set_xlabel('Time t')
    ax[0, 1].set_ylabel('Infected nodes')
    ax[0, 0].set_xlabel('Degree')
    ax[0, 0].set_ylabel('Distribution')

    fig.set_size_inches((8,6))
    ax[1,0].spines['right'].set_visible(False)
    ax[1,0].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].spines['top'].set_visible(False)
    ax[1,1].spines['right'].set_visible(False)
    ax[1,1].spines['top'].set_visible(False)

    plt.tight_layout()
    # plt.savefig('fig2_05-23-22.pdf')
    # plt.savefig('./fig2_04-28-22.png')
    # plt.savefig('./fig2_04-28-22.svg', fmt='svg')
    # fig.savefig('../results/concept_fig2.png')
    # fig.savefig('../results/concept_fig2.svg', fmt='svg')
    # plt.show()

## 7/15
def order_norm_test(A1, A2, beta, orders, norms, ax):
    taus = np.linspace(0.0001, beta*5, 50)  # .7 for example

    xi_order3 = np.zeros(len(taus))
    xi_order2 = np.zeros(len(taus))
    det_temps = np.zeros(len(taus))
    det_temps_halftime = np.zeros(len(taus))
    det_aggs = np.zeros(len(taus))
    det_aggs_halftime = np.zeros(len(taus))
    integral_solutions = np.zeros(len(taus))

    for t, tau in enumerate(taus):
        # Theoretical error
        A_snap = Snapshot(0, tau / beta, beta, A1)
        B_snap = Snapshot(tau / beta, 2 * tau / beta, beta, A2)
        epsilon_combo = Compressor.epsilon(A_snap, B_snap, error_type='combined', order=orders[0], norm=norms[0])
        xi_order2[t] = epsilon_combo
        epsilon_combo = Compressor.epsilon(A_snap, B_snap, error_type='combined', order=orders[1], norm=norms[1])
        xi_order3[t] = epsilon_combo
        # Temporal solution
        y_init = A_snap.dd_normalized
        temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                     networks={tau / beta: A1, 2 * tau / beta: A2})
        solution_t_temporal, solution_p = temp_model.solve_model()
        temporal_timeseries = np.sum(solution_p, axis=0)
        final_temp = temporal_timeseries[-1]
        det_temps[t] = final_temp
        det_temps_halftime[t] = temporal_timeseries[int(len(temporal_timeseries) / 2)]
        # plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
        model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                networks={2 * tau / beta: (A1 + A2) / 2})
        solution_t_agg, solution_p = model.solve_model()
        aggregate_timeseries = np.sum(solution_p, axis=0)
        # plt.show()
        final_agg = aggregate_timeseries[-1]
        det_aggs[t] = final_agg
        det_aggs_halftime[t] = aggregate_timeseries[int(len(aggregate_timeseries) / 2)]

        # Integral
        integrate_between = integrate_error_ts(temporal_ts=solution_t_temporal, temporal_inf=temporal_timeseries,
                                               other_ts=solution_t_agg, other_inf=aggregate_timeseries)
        integral_solutions[t] = integrate_between
    ax[0].plot(taus,xi_order2/ (2 * taus / beta), ls='-.', lw=2, color=type_colors['algo'],
               label=f'Order {orders[0]} Norm {norms[0]}, $\\epsilon(MID)+\\epsilon(END)$')
    ax[0].plot(taus, xi_order3 / (2 * taus / beta), ls='-.', lw=2, color=type_colors['snap2'],
               label=f'Order {orders[1]} Norm {norms[1]}, $\\epsilon(MID)+\\epsilon(END)$')
    ax[0].plot(taus,  (np.abs(det_temps - det_aggs) + np.abs(det_temps_halftime - det_aggs_halftime)), ls='-', lw=2, color='k', alpha=0.6, label='true solution, $\\epsilon_{MID}+\\epsilon_{END}$')
    ax[0].set_xlabel('$\\beta \\cdot \\delta t$')
    ax[0].set_ylabel('Infected nodes')
    ax[0].legend(frameon=False, loc='upper left')

    ax[1].scatter(integral_solutions,xi_order2, color=type_colors['algo'],
                    marker='s', alpha=0.3,
                    label=f'Order {orders[0]} Norm {norms[0]}: increasing $\\beta\\cdot\\delta t$')
    ax[1].scatter(integral_solutions, xi_order3, color=type_colors['snap2'],
                  marker='s', alpha=0.3,
                  label=f'Order {orders[1]} Norm {norms[1]}: increasing $\\beta\\cdot\\delta t$')
    ax[1].set_xlabel('True solution integrated error')
    ax[1].set_ylabel('Error measure $\\xi_{A,B}$')
    ax[1].legend(frameon=False, loc='upper left')
    plt.show()

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
    ax[0,2].hist([sum(A[n]) for n in range(N)],  label='snapshot 1', color='m',alpha=0.5, bins='auto', density=False)
    ax[0,2].hist([sum(B[n]) for n in range(N)],  label='snapshot 2', color='c',alpha=0.5, bins='auto', density=False)
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
    ax2.scatter((np.abs(det_temps-det_aggs) + np.abs(det_temps_halftime-det_aggs_halftime)), error_approx_combo/(2*tau/beta))

    y = np.linspace(0, max(np.abs(det_temps-det_aggs)), 10)
    ax[1, 1].plot(y, y, color='k', ls='--', alpha=0.6)
    y = np.linspace(0, max(np.abs(det_temps_halftime-det_aggs_halftime)), 10)
    ax[1, 0].plot(y, y, color='blue', ls='--', alpha=0.6)
    ax[1, 0].legend()
    ax[1, 1].legend()
    # plt.show()

    #Labels
    ax[0,0].set_xlabel('Time t')
    ax[0,0].set_ylabel('Infected nodes')
    ax[0,0].set_ylabel('Infected nodes')
    ax[0,2].set_xlabel('Degree')
    ax[0,2].set_ylabel('Distribution')
    ax[1, 0].set_xlabel('midpoint difference (true)')
    ax[1, 0].set_ylabel('difference (approx)')
    ax[1, 1].set_xlabel('terminal difference (true)')
    ax[1, 1].set_ylabel('difference (approx)')

    # fig.set_size_inches((10,5))

    plt.tight_layout()
    plt.show()

def monotonic_proof():
    N = 100
    G1, A1 = barbell_graph(N)
    G2, A2 = cycle_graph(N)
    G3, A3 = configuration_model_graph(N)
    G4, A4 = erdos_renyi_graph(N, .01)
    G5, A5 = erdos_renyi_graph(N, .04)
    taus = np.linspace(0.0001, .6, 10)  # .7 for example
    beta = .12

    pairs = [(A1, A2), (A1, A3), (A1, A4), (A1, A5),
             (A2, A1), (A2, A3), (A2, A4), (A2, A5),
             (A3, A2), (A3, A1), (A3, A4), (A3, A5),
             (A4, A2), (A4, A3), (A4, A1), (A4, A5),
             (A5, A2), (A5, A3), (A5, A4), (A5, A1),
             (A2, A2), (A3, A2), (A4, A2), (A5, A2),
             (A2, A3), (A3, A3), (A4, A3), (A5, A3),
             (A2, A4), (A4, A4), (A4, A4), (A5, A4),
             (A2, A5), (A5, A5), (A5, A5), (A5, A5)]
    error_approx_combo = np.zeros((len(pairs), len(taus)))
    integrated_error = np.zeros((len(pairs), len(taus)))
    for p in range(len(pairs)):
        print(p/len(pairs))
        A, B = pairs[p]
        for t, tau in enumerate(taus):
            print(t/len(taus))
            A_lay = Snapshot(0, tau / beta, beta, A)
            B_lay = Snapshot(tau / beta, 2 * tau / beta, beta, B)
            epsilon_combo = Compressor.epsilon(A_lay, B_lay, error_type='combined')
            error_approx_combo[p][t] = epsilon_combo
            # for values tau in 0, T run a deterministic temporal
            y_init = A_lay.dd_normalized
            temp_model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                         networks={tau / beta: A, 2 * tau / beta: B})
            solution_t_temporal, solution_p = temp_model.solve_model()
            temporal_timeseries = np.sum(solution_p, axis=0)
            # plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
            model = TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=2 * tau / beta,
                                    networks={2 * tau / beta: (A + B) / 2})
            solution_t_agg, solution_p = model.solve_model()
            aggregate_timeseries = np.sum(solution_p, axis=0)
            integrate_between = integrate_error_ts(temporal_ts=solution_t_temporal, temporal_inf=temporal_timeseries, other_ts=solution_t_agg, other_inf=aggregate_timeseries)
            integrated_error[p][t] = integrate_between
    return error_approx_combo, integrated_error


# res = monotonic_proof()
# fig, axs = plt.subplots(6, 6)
# ax_y_counter = 0
# for i in range(36):
#     ax_x = i // 6
#     ax_y = ax_y_counter % 6
#     ax = axs[ax_x, ax_y]
#     ax.scatter(res[0][i], res[1][i], color='blue', alpha=(i+4)/40)
#     ax_y_counter += 1
# plt.show()

### 7/15 order and norm experiemnts
# N = 100
# G1, A1 = configuration_model_graph(N)
# G2, A2 = erdos_renyi_graph(N, .012)
# order_norm_test(A1, A2, .12, [1, 3], [None, None], plt.subplots(2)[1])
# order_norm_test(A1, A2, .12, [1, 3], [2, 2], plt.subplots(2)[1])
# order_norm_test(A2, A1, .12, [1, 3], [None, None], plt.subplots(2)[1])
# order_norm_test(A2, A1, .12, [1, 3], [2, 2], plt.subplots(2)[1])
# G1, A1 = barbell_graph(N)
# G2, A2 = cycle_graph(N)
# order_norm_test(A1, A2, .015, [1, 3], [None, None], plt.subplots(2)[1])
# order_norm_test(A1, A2, .015, [1, 3], [2, 2], plt.subplots(2)[1])
# order_norm_test(A2, A1, .015, [1, 3], [None, None], plt.subplots(2)[1])
# order_norm_test(A2, A1, .015, [1, 3], [2, 2], plt.subplots(2)[1])
# plt.show()

###### FIG 2
## 7/7/22: Want to test figure two validations if the absolute value of the matrix elt-wise isn't applied
## Would have to change internal code
N = 100
G3, A3 = configuration_model_graph(N)
G6, A6 = erdos_renyi_graph(N, .012)
taus = np.linspace(0.0001, .6, 50) # .7 for example
A = A3
B = A6

beta = .12
manuscript_fig2(A, B, beta, taus)
plt.savefig('./figures/revisions/fig2_revision.pdf')
plt.show()
# plt.savefig('fig2_05-23-22.pdf')
# quick check with B then A
manuscript_fig2(B, A, beta, taus)
#########


taus = np.linspace(0.0001, .6, 50) # .7 for example
# taus = np.linspace(0.0001, .1, 30) # .7 for example
# taus = np.linspace(0.0001, 2.0, 50) # completely breaks down once network saturates and matexp isn't a good approximation anymore

# concept_custom_durations(A, B, .02, 7, 10) # TODO ok cool idea would be to use the same plot to do multiple time series on the left side
# and then to do a bunch of colored squares on the right that show the dimensions of the approximation
# plt.show()
# concept_custom_durations(A, B, .05, 7, 10)
# concept_custom_durations(A, B, .05, 5, 10)
# concept_custom_durations(B, A, .05, 3, 10)
# concept_custom_durations(B, A, .05, 5, 10)
# plt.show()
# validation_on(A, B, beta, taus)
# manuscript_fig2(A, B, beta, taus)

G3, A3 = cycle_graph(N)
G6, A6 = configuration_model_graph(N)
G3, A3 = erdos_renyi_graph(N, .01)
G4, A4 = erdos_renyi_graph(N, .04)
beta = .06
# try switching order of who is more dense second
# multi_panel_fig_idea2([A6, A3, A3,  A4, A3], beta, [4,6, 11, 14, 16])
multi_panel_fig_idea2([A6, A3, A6, A4], beta, [3,4.5, 7, 9.5])
plt.show()

beta = .12
taus = np.linspace(0.0001, 1.0, 50) # .7 for example
taus = np.linspace(0.0001, 0.5, 50) # .7 for example
# manuscript_fig2(A3, A4, beta, taus)

# validation_on(A6, A3, beta, taus) # less dense first
# validation_on(A3, A6, beta, taus) # less dense second

# Proof of concept for 0-matrix
A0 = np.zeros((N,N))
_, Asparse = erdos_renyi_graph(N, .003)
concept_custom_durations(A, Asparse, .12, 6, 18)
plt.show()
manuscript_fig2(A, A0, beta, taus)

validation_on(A, A0, beta, taus)
validation_on(A0, A, beta, taus)
