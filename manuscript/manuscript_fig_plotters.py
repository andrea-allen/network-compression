import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from matplotlib.ticker import ScalarFormatter, NullFormatter
matplotlib.rcParams.update({'font.size': 12})

colors = sns.color_palette("Paired")
rd_bu = sns.color_palette('RdBu', 30)
icefire = sns.color_palette('icefire', 6)
cmr_map = sns.color_palette('CMRmap_r', 6)
type_colors = {'temp': 'grey', 'even': rd_bu[5], 'algo': rd_bu[27]}
type_colors = {'temp': 'grey', 'even': cmr_map[1], 'algo': cmr_map[5]}

def figure_2():
    print("see concept.py manuscript_fig2()")

def figure_3(all_results):
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3)
    fig.set_size_inches(8, 9)

    results = all_results['one_round_results']
    temp_t = results['temp_t']
    temp_inf = results['temp_inf']
    algo_t = results['algo_t']
    algo_inf = results['algo_inf']
    even_t = results['even_t']
    even_inf = results['even_inf']
    algo_boundary_times = results['algo_boundary_times']
    temp_boundary_times = results['temp_boundary_times']
    even_boundary_times = results['even_boundary_times']
    total_algo_error = results['total_algo_error']
    total_even_error = results['total_even_error']
    total_chosen_error = results['total_chosen_error']

    ######### PANEL (a) (one round time series)
    ax1.plot(temp_t, temp_inf, label='Temporal', color=type_colors['temp'], lw=2.5, alpha=1.0)
    ax1.plot(algo_t, algo_inf, label='Algorithmic', color=type_colors['algo'], lw=2, alpha=1.0, ls='--')
    ax1.plot(even_t, even_inf, label='Even', color=type_colors['even'], lw=2, alpha=1.0, ls='-.')
    ## vertical lines to show compression
    max_infected_buffer = max(max(temp_inf), max(algo_inf)) + 2
    ax1.vlines(temp_boundary_times, ymin=0, ymax=max_infected_buffer / 3, ls='-',
              color=type_colors['temp'], lw=0.5, alpha=1.0)
    ax1.vlines(algo_boundary_times, ymin=2 * max_infected_buffer / 3, ymax=max_infected_buffer,
              ls='--', color=type_colors['algo'], lw=1.5, alpha=0.95)
    ax1.vlines(even_boundary_times, ymin=max_infected_buffer / 3,
              ymax=2 * max_infected_buffer / 3, ls='-.', color=type_colors['even'], lw=1.5, alpha=1.0)
    ax1.set_ylabel('Infected nodes')
    ax1.legend(loc='upper left', frameon=False)

    ############# PANEL (b)
    ax2.plot(algo_t, (-np.array(temp_inf) + np.array(algo_inf)) / np.array(temp_inf),
            # label=f'Algorithmic: {total_algo_error}\n TCE: {total_chosen_error}',
            # label=f'Algorithmic',
            color=type_colors['algo'], ls='--', lw=2, alpha=1.0)
    print(f'Algorithmic: {total_algo_error}\n TCE: {total_chosen_error}')
    ax2.fill_between(algo_t, (-np.array(temp_inf) + np.array(algo_inf)) / np.array(temp_inf),
                    color=type_colors['algo'], alpha=0.5, label='$d_{ALG}$')
    ax2.plot(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf),
            # label=f'Even {total_even_error}',
            # label=f'Even',
            color=type_colors['even'], ls='-.', lw=2)
    print(f'Even {total_even_error}')
    ax2.fill_between(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf), color=type_colors['even'], alpha=0.5, label='$d_{EVEN}$')
    ax2.plot(algo_t, np.zeros(len(algo_t)), color='k', ls='-', alpha=1.0)
    ax2.set_xlabel('Time')
    # ax.set_xlabel('Time (hours)') ## Turn this on for the hospital one
    # ax.set_xticks(np.array(algo_t)[::10])
    # ax.set_xticklabels(np.round((np.array(algo_t)/3600)[::10], 1))
    # ax.set_ylabel('Normalized error')
    ax2.set_ylabel('Normed distance from $x(t)_{TEMP}$')
    ax2.legend(frameon=False, loc='upper right')
    # plt.show()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)


    ########## PANEL (c)
    results = all_results["error_difference_results"]
    iter_range = results["iter_range"]
    optimal_errors_norm = results['opt_error_norm']
    even_errors_norm = results['even_error_norm']
    tce_all = results["tce_all"]
    ax3.scatter(iter_range, optimal_errors_norm/optimal_errors_norm[-1], color=type_colors['algo'], label='$d_{ALG}$', alpha=0.9, s=10)
    ax3.scatter(iter_range, even_errors_norm/even_errors_norm[-1], color=type_colors['even'], label='$d_{EVEN}$', alpha=1.0, s=10)
    ax3.set_xlabel('Resulting number of network snapshots')
    ax3.set_ylabel('Fraction of full error')
    ax3.semilogy()
    ax3.set_yticks([10**(-2), 10**(-1), 10**0])
    ax3.set_yticklabels(['.01', '.1', '1'])

    diffs = optimal_errors_norm / optimal_errors_norm[-1] - even_errors_norm / even_errors_norm[-1]
    diff_colors = ['r' for i in range(len(diffs))]
    for i, d in enumerate(diffs):
        if d < 0:
            diff_colors[i] = type_colors['algo']
        else:
            diff_colors[i] = type_colors['even']

    ax3.vlines(iter_range, optimal_errors_norm/optimal_errors_norm[-1], even_errors_norm/even_errors_norm[-1],
              colors=diff_colors)


    # ax[0].scatter(iter_range, tce_all, color=type_colors['temp'], label='TCE')
    # ax[0].plot(iter_range, tce_all, color=type_colors['temp'], label='TCE')
    # plt.xticks(np.linspace(0, iter_range+1, 10))
    # plt.xticks(list(int(np.arange(0, iter_range+1))))
    # ax3.set_xlabel('Resulting number of network snapshots')
    # ax3.semilogy()
    # ax[0].set_ylabel('Total chosen error')
    # ax3.set_ylabel('Total normalized error')
    x_tick_list = list(iter_range[::4])
    x_tick_list.extend([49])
    ax3.set_xticks(x_tick_list)
    x_tick_labels = list(50 - iter_range[::4])
    x_tick_labels.extend([1])
    x_tick_labels = [int(x) for x in x_tick_labels]
    ax3.set_xticklabels(x_tick_labels)
    # plt.xticks([0, 5, 10, 15, 19])
    plt.legend(loc='upper left', frameon=False)

    # plt.tight_layout()


def figure_4(all_results):
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3)
    fig.set_size_inches(8, 9)

    results = all_results['one_round_results']
    temp_t = results['temp_t']
    temp_inf = results['temp_inf']
    algo_t = results['algo_t']
    algo_inf = results['algo_inf']
    even_t = results['even_t']
    even_inf = results['even_inf']
    algo_boundary_times = results['algo_boundary_times']
    temp_boundary_times = results['temp_boundary_times']
    even_boundary_times = results['even_boundary_times']
    total_algo_error = results['total_algo_error']
    total_even_error = results['total_even_error']
    total_chosen_error = results['total_chosen_error']

    ######### PANEL (b) (one round time series)
    ax2.plot(temp_t, temp_inf, label='Temporal', color=type_colors['temp'], lw=2.5, alpha=1.0)
    ax2.plot(algo_t, algo_inf, label='Algorithmic', color=type_colors['algo'], lw=2, alpha=1.0, ls='--')
    ax2.plot(even_t, even_inf, label='Even', color=type_colors['even'], lw=2, alpha=1.0, ls='-.')
    ## vertical lines to show compression
    max_infected_buffer = max(max(temp_inf), max(algo_inf)) + 2
    ax2.vlines(temp_boundary_times, ymin=0, ymax=max_infected_buffer / 3, ls='-',
              color=type_colors['temp'], lw=0.5, alpha=1.0)
    ax2.vlines(algo_boundary_times, ymin=2 * max_infected_buffer / 3, ymax=max_infected_buffer,
              ls='--', color=type_colors['algo'], lw=1.5, alpha=0.95)
    ax2.vlines(even_boundary_times, ymin=max_infected_buffer / 3,
              ymax=2 * max_infected_buffer / 3, ls='-.', color=type_colors['even'], lw=1.5, alpha=1.0)
    ax2.set_ylabel('Infected nodes')
    ax2.legend(loc='upper left', frameon=False)

    ############# PANEL (b)
    pairwise_error_panel(ax1)

    # panel_b_1(ax2, all_results)
    ax2.set_xlabel('Time')
    ax2.set_xlabel('Time (hours)') ## Turn this on for the hospital one
    # ax2.set_xticks(np.array(algo_t)[::10])
    ax2.set_xticks(3600 * np.array([0, 12, 24, 36, 48, 60, 72, 84, 96]))
    # ax2.set_xticklabels(np.round((np.array(algo_t)/3600)[::10], 1))
    ax2.set_xticklabels([0, 12, 24, 36, 48, 60, 72, 84, 96])


    ########## PANEL (c)
    results = all_results["error_difference_results"]
    # panel_c_idea1(ax3, results, type_colors)
    panel_c_idea2(ax3, results, type_colors)

    print('placeholder debug point')
    # plt.tight_layout()

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

def panel_b_1(ax, all_results):
    results = all_results['one_round_results']
    temp_t = results['temp_t']
    temp_inf = results['temp_inf']
    algo_t = results['algo_t']
    algo_inf = results['algo_inf']
    even_t = results['even_t']
    even_inf = results['even_inf']
    ax.plot(algo_t, (-np.array(temp_inf) + np.array(algo_inf)) / np.array(temp_inf),
            # label=f'Algorithmic: {total_algo_error}\n TCE: {total_chosen_error}',
            # label=f'Algorithmic',
            color=type_colors['algo'], ls='--', lw=2, alpha=1.0)
    ax.fill_between(algo_t, (-np.array(temp_inf) + np.array(algo_inf)) / np.array(temp_inf),
                    color=type_colors['algo'], alpha=0.5, label='$d_{ALG}$')
    ax.plot(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf),
            # label=f'Even {total_even_error}',
            # label=f'Even',
            color=type_colors['even'], ls='-.', lw=2)
    ax.fill_between(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf), color=type_colors['even'], alpha=0.5, label='$d_{EVEN}$')
    ax.plot(algo_t, np.zeros(len(algo_t)), color='k', ls='-', alpha=1.0)
    # ax.set_ylabel('Normalized error')
    ax.set_ylabel('Normed distance from $x(t)_{TEMP}$')
    ax.legend(frameon=False, loc='upper right')
    # plt.show()


def panel_c_idea1(ax, results, type_colors):
    iter_range = results["iter_range"]
    optimal_errors_norm = results['opt_error_norm']
    even_errors_norm = results['even_error_norm']
    ax.scatter(iter_range, optimal_errors_norm[::-1]/optimal_errors_norm[-1], color=type_colors['algo'], label='$d_{ALG}$', alpha=0.6, s=4)
    ax.scatter(iter_range, even_errors_norm[::-1]/even_errors_norm[-1], color=type_colors['even'], label='$d_{EVEN}$', alpha=0.6, s=4)
    ax.set_xlabel('Resulting number of network snapshots')
    ax.set_ylabel('Fraction of final error')
    # ax.set_xticks(iter_range[::4])

    #### IDEA
    axins = ax.inset_axes([.6, .5, 0.3, 0.3])
    # # axins.plot(np.arange(130, 140), np.arange(130,140))
    axins.scatter(iter_range[70:90], optimal_errors_norm[::-1][70:90] / optimal_errors_norm[-1],
                  color=type_colors['algo'], alpha=0.6, s=4)
    axins.scatter(iter_range[70:90], even_errors_norm[::-1][70:90] / even_errors_norm[-1], color=type_colors['even'],
                  alpha=0.6, s=4)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    axins.set_yscale('log')

    #####

    ax.semilogy()
    # ax.semilogx()
    ax.set_xscale('log')
    ax.set_xticks([100, 120, 140, 160, 180, 199])
    ax.set_xticklabels([100, 80, 60, 40, 20, 1])
    ax.set_yticks([10**(-2), 10**(-1), 10**0])
    ax.set_yticklabels(['.01', '.1', '1'])
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.set_xticklabels([100, 80, 60, 40, 20, 1])

    # TODO figure out scales and labels
    # x_tick_list = list(iter_range[::10])
    # x_tick_list.extend([199])
    # ax.set_xticks(x_tick_list)
    # x_tick_labels = list(200 - iter_range[::10])
    # x_tick_labels.extend([1])
    # x_tick_labels = [int(x) for x in x_tick_labels]
    # ax.set_xticklabels(x_tick_labels)

    ax.legend(loc='lower left', frameon=False)

def panel_c_idea2(ax, results, type_colors):
    iter_range = results["iter_range"]
    optimal_errors_norm = results['opt_error_norm']
    even_errors_norm = results['even_error_norm']
    ax.scatter(iter_range, optimal_errors_norm/optimal_errors_norm[-1], color=type_colors['algo'], label='$d_{ALG}$', alpha=0.9, s=5)
    ax.scatter(iter_range, even_errors_norm/even_errors_norm[-1], color=type_colors['even'], label='$d_{EVEN}$', alpha=1.0, s=5)
    ax.set_xlabel('Resulting number of network snapshots')
    ax.set_ylabel('Fraction of full error')
    # ax.set_xticks(iter_range[::4])

    diffs = optimal_errors_norm / optimal_errors_norm[-1] - even_errors_norm / even_errors_norm[-1]
    diff_colors = ['r' for i in range(len(diffs))]
    for i, d in enumerate(diffs):
        if d < 0:
            diff_colors[i] = type_colors['algo']
        else:
            diff_colors[i] = type_colors['even']

    ax.vlines(iter_range, optimal_errors_norm/optimal_errors_norm[-1], even_errors_norm/even_errors_norm[-1],
              colors=diff_colors)

    ### Sorting error classes to show how algorithm performs better
    # for each X, get the even error, E(X).
    # count for how many X can ALG compress to while A(X) < E(X)
    alg_count = np.zeros(len(iter_range))
    for i in range(len(alg_count)):
        Eve_x = even_errors_norm[i]
        j = i
        alg_is_less = True
        while alg_is_less:
            Alg_x = optimal_errors_norm[j]
            if Alg_x < Eve_x:
                alg_count[i] += 1
                j += 1
            if Alg_x > Eve_x:
                alg_is_less = False
            if j == len(alg_count):
                alg_is_less = False #escape at end


    #### INSET
    axins = ax.inset_axes([.1, .55, 0.5, 0.4])
    axins.scatter(iter_range[50:95], (200-iter_range[50:95]) / alg_count[50:95], s=4, color=type_colors['algo'])
    axins.set_xticks([150, 160, 170, 180, 190, 195])
    axins.set_xticklabels([50, 40, 30, 20, 10, 5])
    axins.set_ylim([0,10])
    axins.set_xlabel('number snapshots')
    axins.set_ylabel('ALG factor')
    mean_rate = np.mean((200-iter_range[50:95]) / alg_count[50:95])
    axins.plot(iter_range[50:95], np.full(45, mean_rate), ls="--", color='grey')
    #####

    ax.semilogy()
    # ax.semilogx()
    ax.set_xscale('log')
    ax.set_xticks([100, 120, 140, 160, 180, 199])
    ax.set_xticklabels([100, 80, 60, 40, 20, 1])
    ax.set_yticks([10**(-2), 10**(-1), 10**0])
    ax.set_yticklabels(['.01', '.1', '1'])
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.set_xticklabels([100, 80, 60, 40, 20, 1])

    ax.legend(loc='lower right', frameon=False)


def pairwise_error_panel(ax):
    eps_200 = pd.read_csv("results/pairwise_error_200_layers.csv", index_col=0)
    eps_1000 = pd.read_csv("results/pairwise_error_1000_layers.csv", index_col=0)
    eps_4000 = pd.read_csv("results/pairwise_error_4000_layers.csv", index_col=0)
    ax.scatter(eps_4000["st_times"], eps_4000["pariwise_eps"]/max(eps_4000["pariwise_eps"]), label= "4000", s=12, alpha=0.5, color=cmr_map[0],
               edgecolor='face')
    ax.scatter(eps_1000["st_times"], eps_1000["pariwise_eps"]/max(eps_1000["pariwise_eps"]), label= "1000", s=12, alpha=0.5, color=cmr_map[3],
               edgecolor='face')
    ax.scatter(eps_200["st_times"], eps_200["pariwise_eps"]/max(eps_200["pariwise_eps"]), label= "200", s=12, alpha=1.0, color=cmr_map[5],
               edgecolor='face')
    ax.set_ylabel("Relative pairwise error $\\xi_{A,B}$")
    ax.legend(loc='upper left', frameon=False)
    return ax


"""
SYNTHETIC PLOT (FIGURE 3)
"""
results_one_round = json.load(open('./results/synthetic_data.json', "r"))
result_array = np.loadtxt('./results/synthetic_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
figure_3({"one_round_results": results_one_round, "error_difference_results": results})
plt.tight_layout()
plt.show()

"""
HOSPITAL DATA PLOT (FIGURE 4)
"""
results_one_round = json.load(open('results/hospital_data.json', "r"))
result_array = np.loadtxt('./results/hospital_error_integrals.txt', delimiter=',')
results = {'iter_range': result_array[0], 'even_error_norm': result_array[1], 'opt_error_norm': result_array[2], 'tce_all': result_array[3]}
figure_4({"one_round_results": results_one_round, "error_difference_results": results})
plt.tight_layout()
plt.show()

